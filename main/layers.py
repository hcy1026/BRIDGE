from typing import List, Union, Tuple, Dict, Optional

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BaseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.w1(x))
        return self.w2(self.dropout(h))


class ResNet1DLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        self.ff = BaseMLP(hidden_size, intermediate_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff(x) + x
        x = self.norm(x)
        return x


class ResNet1DBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResNet1DLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class FusionEncoder(nn.Module):
    """
    External interface stays unchanged.

    Example external proj_meta:
    {
        "CLIP-ViT-B-32-laion2B-s34B-b79K": 4352,
    }

    Internally:
    - packed CLIP multi-scale branch -> split into sub-branches
      then each sub-branch -> Linear(raw_dim, branch_dim)
      then concat all projected branches
      then big Linear(total_projected_dim, hidden_size)
      then ResNet1DBlock
    """
    DEFAULT_PACKED_SPECS = {
        "CLIP-ViT-B-32-laion2B-s34B-b79K": {
            "subkeys": [
                "hs_2",
                "hs_5",
                "hs_8",
                "hs_11",
                "final",
            ],
            "subdims": [768, 768, 768, 768, 512],
        },
    }

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 1024,
        proj_meta: Optional[Dict[str, int]] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        branch_dim: int = 256,
        use_branch_norm: bool = False,
        use_branch_act: bool = False,
        use_pre_norm: bool = False,
        packed_specs: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ):
        super().__init__()

        proj_meta = proj_meta or {}
        if len(proj_meta) == 0:
            raise ValueError("FusionEncoder requires non-empty proj_meta.")

        self.input_meta = dict(proj_meta)
        self.input_keys = list(proj_meta.keys())
        self.branch_dim = int(branch_dim)

        self.packed_specs = dict(self.DEFAULT_PACKED_SPECS)
        if packed_specs is not None:
            self.packed_specs.update(packed_specs)

        self.expanded_branches: List[Dict[str, object]] = []
        self.branch_order: List[str] = []

        for parent_key in self.input_keys:
            raw_dim = int(proj_meta[parent_key])

            if parent_key in self.packed_specs:
                spec = self.packed_specs[parent_key]
                subkeys = spec["subkeys"]
                subdims = spec["subdims"]

                if sum(subdims) != raw_dim:
                    raise ValueError(
                        f"Packed spec mismatch for '{parent_key}': "
                        f"proj_meta says {raw_dim}, but subdims sum to {sum(subdims)}"
                    )

                start = 0
                for subkey, subdim in zip(subkeys, subdims):
                    end = start + int(subdim)
                    branch_name = f"{parent_key}::{subkey}"
                    self.expanded_branches.append(
                        {
                            "parent_key": parent_key,
                            "subkey": subkey,
                            "start": start,
                            "end": end,
                            "dim": int(subdim),
                            "branch_name": branch_name,
                        }
                    )
                    self.branch_order.append(branch_name)
                    start = end
            else:
                branch_name = f"{parent_key}::full"
                self.expanded_branches.append(
                    {
                        "parent_key": parent_key,
                        "subkey": "full",
                        "start": 0,
                        "end": raw_dim,
                        "dim": raw_dim,
                        "branch_name": branch_name,
                    }
                )
                self.branch_order.append(branch_name)

        self.branch_proj = nn.ModuleDict()
        for branch in self.expanded_branches:
            branch_name = branch["branch_name"]
            in_dim = int(branch["dim"])

            layers = [nn.Linear(in_dim, self.branch_dim)]
            if use_branch_norm:
                layers.append(nn.LayerNorm(self.branch_dim))
            if use_branch_act:
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            self.branch_proj[self._safe_name(branch_name)] = nn.Sequential(*layers)

        fused_dim = len(self.expanded_branches) * self.branch_dim
        pre_norm = nn.LayerNorm(fused_dim) if use_pre_norm else nn.Identity()

        self.input_proj = nn.Sequential(
            pre_norm,
            nn.Linear(fused_dim, hidden_size),
        )

        self.block = ResNet1DBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            num_layers=num_layers,
        )

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", name)

    def _split_parent_input(
        self,
        parent_key: str,
        emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Split one external input into internal branches if needed.
        """
        if emb.ndim > 2:
            bsz = emb.shape[0]
            emb = emb.reshape(bsz, -1)

        out = {}

        related = [b for b in self.expanded_branches if b["parent_key"] == parent_key]
        expected_dim = sum(int(b["dim"]) for b in related)

        if emb.shape[-1] != expected_dim:
            raise ValueError(
                f"Input dim mismatch for '{parent_key}': "
                f"expected {expected_dim}, got {emb.shape[-1]}"
            )

        for branch in related:
            s = int(branch["start"])
            e = int(branch["end"])
            branch_name = branch["branch_name"]
            out[branch_name] = emb[:, s:e]

        return out

    def forward(self, **x: torch.Tensor):

        missing = [k for k in self.input_keys if k not in x]
        if missing:
            raise KeyError(
                f"Missing input keys {missing}. "
                f"Expected external keys: {self.input_keys}, got: {list(x.keys())}"
            )

        split_dict: Dict[str, torch.Tensor] = {}
        for parent_key in self.input_keys:
            parent_emb = x[parent_key]
            split_dict.update(self._split_parent_input(parent_key, parent_emb))

        projected = []
        for branch_name in self.branch_order:
            proj = self.branch_proj[self._safe_name(branch_name)]
            branch_emb = split_dict[branch_name]
            projected.append(proj(branch_emb))   # [B, branch_dim]

        x = torch.cat(projected, dim=-1)         # [B, num_branches * branch_dim]

        x = self.input_proj(x)                   # [B, hidden_size]

        x = self.block(x)                        # [B, hidden_size]

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, c_num: int = 17):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, c_num=17, **kwargs):
        super().__init__(PatchEmbedding(emb_size, c_num), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )
