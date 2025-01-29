# Data

We maintain the data collection in [this repo at HuggingFace Hub Platform](https://huggingface.co/datasets/Qingyun/lmmrotate-sft-data), which hosts the training&evaluation data from five publicly-available datasets comprising **DOTA-v1.0**, **DIOR-R**, **FAIR1M-v1.0**, **SRSDD**, **RSAR**.

To make it easier for users to download and use, we have uploaded all the processed images and annotations. 
We recommend downloading the entire dataset and extracting it, as we strive to make it ready for use immediately. 
If you already have some of the images or annotations locally, you can exclude certain files during the download to save time. 
We have compressed the images and annotations for each dataset separately to facilitate such convenience.

If you encounter any issues, such as errors in the data or have other questions about the dataset, feel free to contact me via [GitHub issues](https://github.com/Li-Qingyun/mllm-mmrotate/issues)(prefered), or email me directly. 
I will continue to maintain the dataset.

## Downloading Guide

You can download with your web browser on [the file page](https://huggingface.co/datasets/Qingyun/lmmrotate-sft-data/tree/main).

We recommand downloading in terminal using huggingface-cli (`pip install --upgrade huggingface_cli`). You can refer to [the document](https://huggingface.co/docs/huggingface_hub/guides/download) for more usages.

```
# Set Huggingface Mirror for Chinese users (if required):
export HF_ENDPOINT=https://hf-mirror.com 
# Download the whole folder (you can also modify local-dir with your data path and make soft link here):
huggingface-cli download Qingyun/lmmrotate-sft-data --repo-type dataset --local-dir mllm-mmrotate/playground/data
# If any error (such as network error) interrupts the downloading, you just need to execute the same command, the latest huggingface_hub will resume downloading.
```

If you already download some data, you can also exclude them to save time. For example, you can exclude DOTA(split_ss_dota) trainval images with the `--exclude` option. You can also only download certain file with the position arg `filenames` or the `--include` option.

```
# This will exclude the files and just download the others.
huggingface-cli download Qingyun/lmmrotate-sft-data --repo-type dataset --local-dir mllm-mmrotate/playground/data --exclude **split_ss_dota_trainval**
# This will download the file and should put it in the folder.
huggingface-cli download Qingyun/lmmrotate-sft-data split_ss_dota/trainval/split_ss_dota_trainval_annfiles.tar.gz --repo-type dataset --local-dir mllm-mmrotate/playground/data
# This will download the files and put them like the arrangement in the repo.
huggingface-cli download Qingyun/lmmrotate-sft-data --repo-type dataset --local-dir mllm-mmrotate/playground/data --include **split_ss_dota_trainval**
```

Then, extract all files from the compressed files.

```
find . -name "*.tar.gz" -execdir tar -zxvf {} \;
```

At last, if required, you can delete all the compressed files.
```
# list the files to delete for checking (if required)
find . -type f -name "*.tar.gz*" -print
# delete
find . -type f -name "*.tar.gz*" -exec rm -f {} \;
```

## Explanation

- `split_ss_dota`: The simplest `split_ss_dota` prepared following [official instruction in MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data/dota). We add `train` and `val` folders, which only have individual light `annfiles` folders and share `images` folder with `trainval` split.
- `split_ss_fair1m_1_0` and `split_ss_fair1m_2_0`: We just use script in [this csdn blog](https://blog.csdn.net/weixin_45453121/article/details/132224388) to convert the fair1m xml files into DOTA format to reuse tools of DOTA. We then follow [whollywood](https://github.com/yuyi1005/whollywood) to prepare the dataset. NOTE that the differences between FAIR1M-v1.0 and FAIR1M-v2.0 are (Hence, FAIR1M-v1.0 and FAIR1M-v2.0 share the `train` split):
>  Compared with 1.0, validation sets have been added in FAIR1M 2.0, and test set have been expanded. The train set of FAIR1M-1.0 and FAIR1M-2.0 are consistent. ([Description](https://gaofen-challenge.com/benchmark
- `DIOR-R`: The simplest `dior` prepared following [official instruction in MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data/dior). We concat ImageSets/Main/train.txt and ImageSets/Main/val.txt into ImageSets/Main/trainval.txt, so you do not need to use the ConcatDataset for `trainval` split.
- `SRSDD`: The simplest `srsdd` prepared following [official instruction in MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data/dior). We also add json format annotation prepared with [this script](https://huggingface.co/datasets/Qingyun/lmmrotate-sft-data/blob/main/SRSDD/convert_ann_to_json.py).
- `RSAR`: The simplest `rsar` prepared following [official instruction in RSAR](https://github.com/zhasion/RSAR?tab=readme-ov-file#2-dataset-prepare). **You need to run [this script](https://huggingface.co/datasets/Qingyun/lmmrotate-sft-data/blob/main/RSAR/symlink_creator.py) to prepare `trainval` folder.**
```bash
python -u RSAR/symlink_creator.py RSAR/train/images RSAR/trainval/images
python -u RSAR/symlink_creator.py RSAR/val/images RSAR/trainval/images
python -u RSAR/symlink_creator.py RSAR/train/annfiles RSAR/trainval/annfiles
python -u RSAR/symlink_creator.py RSAR/val/annfiles RSAR/trainval/annfiles
```

## Statement and ToU

We release the data under a CC-BY-4.0 license, with the primary intent of supporting research activities. 
We do not impose any additional using limitation, but the users must comply with the terms of use (ToUs) of the source dataset. 
This dataset is a processed version, intended solely for academic sharing by the owner, and does not involve any commercial use or other violations of the ToUs. 
Any usage of this dataset by users should be regarded as usage of the original dataset. 
If there are any concerns regarding potential copyright infringement in the release of this dataset, please contact me, and I will remove any data that may pose a risk.

## Cite

LMMRotate paper:
```
@article{li2025lmmrotate,
  title={A Simple Aerial Detection Baseline of Multimodal Language Models},
  author={Li, Qingyun and Chen, Yushi and Shu, Xinya and Chen, Dong and He, Xin and Yu Yi and Yang, Xue },
  journal={arXiv preprint arXiv:2501.09720},
  year={2025}
}
```

Please also cite the paper of the original source dataset if they are adopted in your research.
