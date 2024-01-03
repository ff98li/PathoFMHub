# PathoFMHub
This repository contains the codebase for Benchmarking the utilities of pathology foundation models in Whole Slide Images (WSI) analysis.
The dataset is from the [UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN) competition - 2023](https://www.kaggle.com/competitions/UBC-OCEAN) for classification of five ovarian cancer subtypes from histopathology whole slide images (WSI) and tissue microarray (TMA).

## Requirements

The codebase is developed with:

- `Gentoo 2.6` | `Python 3.9.6` | `CUDA 11.7` | `Pytorch 2.0.1`


## Dependencies and installation

This codebase is based on IBMIL and DSMIL. It assumes that you have a working virtual environment (e.g. `conda` or `virtualenv`) with Python 3.9 and Pytorch 2.0.1 installed.

Please install the required dependencies

```bash
pip install -r requirements.txt
```

### IBMIL

IBMIL requires faiss for clustering in interventional training. 

Faiss provides official precompiled libraries for Anaconda in Python:
[faiss-cpu](https://anaconda.org/pytorch/faiss-cpu) and [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu).

For computing clusters where conda is not available, please consult the the technical documentation of your clusters, or consider [building from source](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

### Data

Ovarian cancer biopsy WSI and TMA dataset can be downloaded from the UBC Ovarian Cancer subtypE clAssification and outlier detectioN (UBC-OCEAN) competition on [Kaggle](https://www.kaggle.com/competitions/UBC-OCEAN/data).

## Examples

### Preprocess

#### Generate tile JSON lists for each WSI

```bash
python gen_patch_json.py
```
The script assumes the dataset is downloaded and extracted in the `./UBC-OCEAN` folder. If it's not the case, please modify the data path with `--data_root` accordingly. Once the script is finished, it will generate a `patch.json` file in the current working directory.

The `patch.json` file is a list of selected patches'coordinates for each pathology image. The selection criteria is based on the tissue percentage of each patch as used in PLIP. The tissue percentage is computed by the ratio of the number of non-white/non-black pixels (RGB > 10 and <= 200) to the total number of pixels in each patch. The threshold is set to 0.5.

#### Generate bag embeddings

PLIP's pretrained ViT is used as the embedder for patch-level feature extraction:

```bash
python compute_feats_plip.py \
    --num_classes 6 \
    --batch_size 32 \
    --num_workers 4 \
    --backbone plip \
    --dataset UBC-OCEAN \
    --patch_ext png \
    --json_path patch.json \
```

### Train

#### Train aggregator

```bash
python train_ubc_ocean.py \
    --num_classes 5 \ ## number of non-negative classes
    --feats_size 512 \
    --dataset UBC-OCEAN \
    --model dsmil \
    --num_epochs 50 \
    --agg no \
    --seed 2023 \
    --agg_save_dir baseline_dsmil \
    --lr 0.0005
```

#### Generate confounder

```bash
python clustering.py \
    --num_classes 5 \ ## non-negative classes
    --feats_size 512 \
    --dataset UBC-OCEAN \
    --model dsmil \
    --load_path path/to/trained/aggregator/pth \
```

#### Interventional training

```bash
python train_ubc_ocean.py \
    --num_classes 5 \ ## non-negative classes
    --feats_size 512 \
    --dataset UBC-OCEAN \
    --model dsmil \
    --num_epochs 50 \
    --agg no \
    --seed 2023 \
    --lr 0.0005 \
    --agg_save_dir dsmil \
    --c_path datasets_deconf/UBC-OCEAN/train_bag_cls_agnostic_feats_proto_<k>.npy
```
where `<k>` is the number of clusters.

### Inference

```bash
python test_ubc_ocean.py \
    --num_classes 5 \ ## non-negative classes
    --feats_size 512 \
    --dataset UBC-OCEAN \
    --model dsmil \
    --num_epochs 50 \
    --agg no \
    --seed 2023 \
    --c_path datasets_deconf/UBC_OCEAN/train_bag_cls_agnostic_feats_proto_<k>.npy \
    --agg_save_dir dsmil \
    --load_path deconf_dsmil/path/to/deconf/dsmil \
    --test \
    --lr 0.0005
```

## References

* Huang, Z., Bianchi, F., Yuksekgonul, M., Montine, T. J., Zou, J.: A visual–language foundation model for pathology image analysis using medical Twitter. Nature medicine, 29(9), 2307-2316 (2023)

* Li, B., Li, Y., Eliceiri, K. W.: Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14318-14328 (2021)

* Lin, T., Yu, Z., Hu, H., Xu, Y., Chen, C. W.: Interventional bag multi-instance learning on whole-slide pathological images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19830-19839 (2023)