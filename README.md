# BRIDGE: Brain-Vision Representation Integration through Depth and Granularity Encoding

This repository is the official implementation of BRIDGE: Brain-Vision Representation Integration through Depth and Granularity Encoding.
<!--(https://arxiv.org/abs/2030.12345)-->

BRIDGE, a brain-vision representation integration framework that aligns two modalities through Visual Depth Encoding and Brain Granularity Encoding. BRIDGE extracts and fuses LAION CLIP ViT-B-32 representations from multiple depths, explicitly partitions stimulus-evoked EEG/MEG responses into a small number of temporally ordered stages. The resulting brain and visual embeddings are trained in a shared latent space by contrastive learning and can further support brain-to-image generation through a pretrained diffusion prior. 

# Get Started

This repo provides four main stages:

1) **Build visual embeddings**
2) **Train brain-vision contrastive model**
3) **Train prior**
4) **Build reconstruction**

## 0. Environment

```bash
git clone git@github.com:ssshamiii/Brain-HIVE.git
cd BRIDGE-main

conda create -n bridge python=3.13 -y
conda activate bridge

pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## 1. Prepare data & pretrained weights

### Datasets

The scripts assume the following layout:

```text
<BASE_DIR>/
  data/
    things-eeg/
      Preprocessed_data_250Hz_whiten/   # EEG numpy / mat files
      ...                               # THINGS images live under this folder
      embeddings/                       # output from build_embeddings
    things-meg/
      Preprocessed_data/                # MEG numpy / mat files
      ...
      embeddings/
    visual-layer/
      imagenet-1k-vl-enriched/
        data/                           # ImageNet images
        embeddings/                     # output from build_embeddings
```
Data sources:
  - EEG: [things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg)
  - MEG: [things-meg](https://huggingface.co/datasets/Haitao999/things-meg)
  - ImageNet: [imagenet-1k-vl-enriched](https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched)
### Pretrained diffusion (SDXL) + IP-Adapter weights

Prior training uses SDXL-base and inference uses SDXL-turbo for fast sampling 
Requirements:

- **SDXL base** (HuggingFace ID): `stabilityai/stable-diffusion-xl-base-1.0`
- **SDXL turbo** (HuggingFace ID): `stabilityai/sdxl-turbo`
- **SDXL IP-Adapter weights**: a single `*.safetensors` file: `h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.safetensors`

We recommend storing them under:

```text
<BASE_DIR>/
  pretrained/
    ip-adapter_sdxl.safetensors
    stable-diffusion-xl-base-1.0
    ...
  priors/              # output of train_prior.sh
```

### Visual encoders

`build_embeddings.sh` can build multistage embeddings from pretrained visual CLIP ViT-B/32 - LAION-2B encoder. By default it looks under:

```text
<BASE_DIR>/pretrained/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

If you prefer to load from HuggingFace directly from `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` , set `MODEL_PATH` in `scripts/build_embeddings.sh` to the HF IDs instead of local files.

## 2. Build visual embeddings

```bash
bash scripts/build_embeddings.sh
```

## 3. Train brain-vision contrastive model

```bash
# Intra-subject
bash scripts/train_clip_intra.sh

# Inter-subject
bash scripts/train_clip_inter.sh
```

## 4. Train prior

```bash
bash scripts/train_prior.sh
```
## 5. Build reconstruction

Reconstruction uses a pretrained Prior.

```bash
bash scripts/build_reconstruction.sh
```
<!--
# Citation

If you find our project is helpful, please cite our paper as

```

```
-->

# Related Work

Similar works...

[Learning Brain Representation with Hierarchical Visual Embeddings](https://openreview.net/forum?id=IEq71qS8B7)

[Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](https://arxiv.org/abs/2503.04207)

[Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba5f1233efa77787ff9ec015877dbd1f-Abstract-Conference.html)


## Results

Our model achieves the following performances on :

### Brain-to-Image Retrieval on THINGS-EEG and THINGS-MEG

| Dataset  | Intra Top 1 Accuracy  | Intra Top 5 Accuracy | Inter Top 1 Accuracy | Inter Top 5 Accuracy |
|----------|-----------------------| -------------------- | -------------------- | -------------------- |
|THINGS-EEG|          80.8%        |      97.0%           |         33.7%        |      65.8%           |
|THINGS-MEG|          33.1%        |      61.5%           |         7.4%         |      19.3%           |

<!--
## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. -->
