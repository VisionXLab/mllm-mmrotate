<p align="center">
  <h1 align="center">LMMRotate 🎮: A Simple Aerial Detection Baseline of Multimodal Language Models</h1>
  <p align="center">
      <a href='https://scholar.google.com/citations?hl=en&user=TvsTun4AAAAJ' style='text-decoration: none' >Qingyun Li</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=A39S7JgAAAAJ&hl=en' style='text-decoration: none' >Yushi Chen</a><sup></sup>&emsp;
      <a href='https://www.researchgate.net/profile/Shu-Xinya' style='text-decoration: none' >Xinya Shu</a><sup></sup>&emsp;      
      <a href='https://scholar.google.com/citations?hl=en&user=UzPtYnQAAAAJ' style='text-decoration: none' >Dong Chen</a><sup></sup>&emsp;     
      <a href='https://scholar.google.com/citations?hl=en&user=WQgE8l8AAAAJ' style='text-decoration: none' >Xin He</a><sup></sup>&emsp;     
      <a href='https://scholar.google.com/citations?user=OYtSc4AAAAAJ&hl=en' style='text-decoration: none' >Yi Yu</a><sup></sup>&emsp;     
      <a href='https://yangxue0827.github.io/' style='text-decoration: none' >Xue Yang</a><sup></sup>&emsp;
      <div align="center">
      <a href='https://arxiv.org/abs/2501.09720'><img src='https://img.shields.io/badge/arXiv-2501.09720-brown.svg?logo=arxiv&logoColor=white'></a>
      <a href='https://huggingface.co/collections/Qingyun/lmmrotate-6780cabaf49c4e705023b8df'><img src='https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=HuggingFace&logoColor=white'></a>
	  </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

This repo is a technical practice to fine-tune **L**arge **M**ultimodal language **M**odels for oriented object detection as in [MMRotate](https://github.com/open-mmlab/mmrotate) and hosts the official implementation of the paper: **A Simple Aerial Detection Baseline of Multimodal Language Models**.

<img src="https://github.com/user-attachments/assets/d34e4c0c-9e04-446e-a511-2e7005e32074" alt="framework" width="100%" />

We currently support fine-tuning and evaluating [Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de) models on three optical datasets (DOTA-v1.0, DIOR-R, FAIR1M-v1.0) and two SAR datasets (SRSDD, RSAR) as reproductions of experimental results in the [technical report paper](https://arxiv.org/abs/2501.09720). Thanks to the strong grounding and detection performance of the pre-trained foundation model, our [detection performance](https://github.com/user-attachments/assets/2f45fad2-bab9-45f3-8b7f-fdd1a16db335) rivals conventional detectors (e.g., RetinaNet, FCOS), even in challenging scenarios with dense and small-scale objects in the images. We hope that this baseline will serve as a reference for future MLM development, enabling more comprehensive capabilities for understanding remote sensing data.

## Performance

Get [**model weight**](https://huggingface.co/collections/Qingyun/lmmrotate-6780cabaf49c4e705023b8df) on [Huggingface](https://huggingface.co/collections/Qingyun/lmmrotate-6780cabaf49c4e705023b8df)

[**Click here**](https://github.com/user-attachments/assets/2f45fad2-bab9-45f3-8b7f-fdd1a16db335) for the visualization of the MLM detector, you can zoom in for a clearer view.

<img src="https://github.com/user-attachments/assets/dcc2353f-4060-40e1-95d7-f926242691b2" alt="framework" width="80%" />

The `mAP_nc` represents 'mAP without confidence score'. As our detector does not output confidence score, we use mAP_nc and mF_1 as evaluation metrics. You can refer to the [technical report paper](https://arxiv.org/abs/2501.09720) for more details. [This notebook](https://github.com/Li-Qingyun/mllm-mmrotate/blob/master/playground/evaluate_without_scores.ipynb) provides the practices during exploring stage.

## Get Started

First, refer to [Enviroment.md](Enviroment.md) to prepare an enviroment.

Then, refer to [Data.md](Data.md) to prepare/download the data.

> NOTE:
> 1. We support multi-nodes distributed training based on SLURM. If your resource platform is different and requires multi-nodes distributed training, you may need adapt the shell scripts to your platform. Or you can mult the node count to gradient_accumulation_steps option. Concat us in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) for more support.
> 2. The v2 in script name to record data version is response format version, not dataset version. `dota1-v2` means DOTA-v1.0 of 2-th response.
> 3. The users may misunderstand the data split name. We use `trainval` to represent all the default training split (training with `trainval` if `val` exist, else `train` only. testing with `test` only). However, as is described in the paper, the mF1 calculation requires ground-truth for evaluation. Hence, we add `-train` behind the dataset name to indicate only using `train` for training and `val` for evaluation. (Contact me in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) if there are still confusing things. I paint a pie to refactor this in future.)

### Practices

- train an aerial detector based on [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) on DOTA-v1.0:
```shell
# You can train the model on a standalone with 4xRTX4090:
bash scripts/florence-2-l_vis1024-lang2048_dota1-v2_b2x4xga4-50e.sh
# If you have multi-node cluster based on Slurm, you can also train the model on 16 gpus:
srun -p <your slurm partition> --job-name=lmmrotate-dota-train \
    --gres=gpu:8 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=96 \
    --kill-on-bad-exit=1 --quotatype=reserved \
    bash scripts/florence-2-l_vis1024-lang2048_dota1-v2_b2x16-100e.sh
```
- inference and get mAP_nc on DOTA-v1.0:
```shell
# Using single gpu:
bash scripts/eval_standalone.sh <checkpoint folder path>
# For a standalone with 4xRTX4090:
NGPUS=4 bash scripts/eval_standalone.sh <checkpoint folder path>
# For Slurm users:
srun -p <your slurm partition> --job-name=lmmrotate-dota-eval \
    --gres=gpu:8 --ntasks=<nodes number> --ntasks-per-node=1 --cpus-per-task=96 \
    --kill-on-bad-exit=1 --quotatype=reserved \
    bash scripts/eval_slurm.sh <checkpoint folder path>
```

- get f1:
```shell
# then get f1
python -u -m lmmrotate.modules.f1_metric <checkpoint folder path>/<pkl file>
```

- visualization (for sampled 20 figures)
```shell
bash scripts/eval_standalone.sh <checkpoint folder path> --shuffle_seed 42 --clip_num 20 --vis
```

- train a baseline detector and get the map, map_nc, and f1 score (take Rotated-RetinaNet@RSAR as an example)
```shell
# train the detector
python -u playground/mmrotate_train.py playground/mmrotate_configs/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024.py --work-dir playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024
# inference on test dataloader to get result pickle file
python -u playground/mmrotate_test.py playground/mmrotate_configs/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024.py playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024/epoch_12.pth --out playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024/results_test.pkl
# get map_ac (best of map_nc on a sequence of score thresholds)
python -u playground/eval_mmrotate_detector_mapnc.py --dataset_name rsar --pickle_result_path playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024/results_test.pkl
# get f1 (best of f1 on a sequence of score thresholds)
python -u -m lmmrotate.modules.f1_metric playground/mmrotate_workdir/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar-1024/results_test.pkl
```

### Interface

Some options of the [training script](lmmrotate/train.py):

- `data_path` and `image_folder`: you can pass multiple data here to run multiple dataset, remember to set dataset_mode
- `dataset_mode`: optional from `single`, `concat`, and `balanced`, refer to the paper for more details.
- `model_type`: optional from `florence2` now, more models are working in progress.
- `model_name_or_path`: provide the pretrained model path or name on huggingface hub.
- `image_square_length`: set 1024 to train `florence2` on 1024x1024, and it is useless for models with dynamic resolutions.
- `language_model_max_length`: model_max_length for the language model.
- `model_revision`: commit id of the model on huggingface hub. (Florence-2-large had an update, which is not used in this repo.)
- `language_model_lora`: lora option similar to internvl.
- `response_format`: box encoding and decoding format.
- ...... (Contact me in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) if there are questions.)

Some options of the [map_nc eval script](lmmrotate/eval.py):

- `model_ckpt_path`: checkpoint path, you can pass multiple ckpt
- `result_path`: folder to save eval log
- `eval_intermediate_checkpoints`: whether to eval intermediate checkpoints
- `vis`: visualize the result while evaluating
- `pass_evaluate`: only inference and dump results, do not evaluate the result. (because inference requires gpu, while evaluate does not)
- `dataset_type`: which dataset to eval. if is not passed, it will be decided according to checkpoint name.
- `split`: which splits to get evaluation results.
- `clip_num`: when you want to get results fast or visualize the results, you can clip the dataset.
- `shuffle_seed`: seed for clip dataset.

## Contact and Acknowledge

Feel free to contact me through my email (21b905003@stu.hit.edu.cn) or [github issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues). I'll continue to maintain this repo.

The code is based on [MMRotate](https://github.com/open-mmlab/mmrotate) and [Transformers](https://github.com/huggingface/transformers). Many modules refer to [InternVL](https://github.com/OpenGVLab/InternVL) and [LLaVA](https://github.com/haotian-liu/LLaVA). The model architecture benefits from the open-source general-purpose vision-language model [Florence-2](https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de). Thanks for their brilliant works.

## Citation

If you find our paper or benchmark helpful for your research, please consider citing our paper and giving this repo a star ⭐. Thank you very much!

```bibtex
@article{li2025lmmrotate,
  title={A Simple Aerial Detection Baseline of Multimodal Language Models},
  author={Li, Qingyun and Chen, Yushi and Shu, Xinya and Chen, Dong and He, Xin and Yu Yi and Yang, Xue },
  journal={arXiv preprint arXiv:2501.09720},
  year={2025}
}
```
