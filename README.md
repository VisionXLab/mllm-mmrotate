# lmmrotate

We are still working in progress~

[ArXiv]() | [ResearchGate (Full Text Available)]() | [Huggingface]()

This repo aims to fine-tune **L**arge **M**ultimodal language **M**odels for rotated object detection as in [MMRotate](https://github.com/open-mmlab/mmrotate) and hosts the official implementation of the paper: **A Simple Aerial Detection Baseline of Multimodal Language Models**, _ArXiv 25xx.xxxxx_, Qingyun Li, Yushi Chen, Xinya Shu, Dong Chen, Xin He, Yi Yu, and Xue Yang.

<img src="https://github.com/user-attachments/assets/cf18cc4c-d2d4-490b-87c0-3956d3b7cb9b" alt="framework" width="100%" />

## Performance

[Click here](https://github.com/user-attachments/assets/f61edcd2-1dee-4bdb-8a1e-c8dd1cf163a1) for the visualization of the MLM detector.

## Enviroment
**NOTE: a misaligned enviroment between inference and training may cause bad effect.**

- create env and install torch
```shell
conda create -n lmmrotate python=3.10.12
conda activate lmmrotate
```

- set cuda&gcc (recommanded for current enviroment, you can also set it in ~/.bashrc)
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
touch $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
vim $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
```
write the following lines
```shell
# set cuda&gcc home
export CUDA_HOME=todo  # change this to <path to cuda-12.1>
export GCC_HOME=todo  # change this to <path to gcc (such as 10.1)>
# remove redundant cuda&gcc path
export PATH=$(echo "$PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's#[^:]*cuda[^:]*:##g' -e 's#:[^:]*cuda[^:]*##g' -e 's#[^:]*gcc[^:]*:##g' -e 's#:[^:]*gcc[^:]*##g')
# set cuda&gcc path
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:$LD_LIBRARY_PATH
# set site-packages path
export SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
```
then `conda activate lmmrotate` to enable these env vars

- install torch
```shell
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

- build and install [mmcv](https://mmcv.readthedocs.io/en/latest/)
```shell
# install mim
pip install openmim

# install mmcv
# install with openmim
mim install "mmcv==2.0.1"
# install from source (recommanded)
git clone https://github.com/open-mmlab/mmcv.git $SITE_PACKAGES_PATH/mmcv
cd $SITE_PACKAGES_PATH/mmcv
git checkout v2.0.1
pip install -r requirements/optional.txt
echo 'set -x;TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print(f'\''{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}'\'')") pip install -e . -v' >> install.sh
bash install.sh
```
The compiling of mmcv-v2.0.1 may raise error, because torch require C++17 or later compatible compiler. One solution is in [this issue](https://github.com/open-mmlab/mmcv/issues/2860).
> Changing `c++14` to `c++17` in [the 204 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L204) and [the 421 line](https://github.com/open-mmlab/mmcv/blob/d28aa8a9cced3158e724585d5e6839947ca5c449/setup.py#L421) of the `setup.py` can temporarily fix this issue.

- install openmmlab mmdet and mmrotate
```shell
mim install "mmdet==3.0.0"
mim install "mmrotate==1.0.0rc1"
```

- install [flash-attention](https://github.com/Dao-AILab/flash-attention)
```shell
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

- install lmmrotate
```shell
pip install -e .
```

- The torch-2.3.0 may raise a warning as:
> site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.) In version 2.3.0 of pytorch, it prints this unwanted warning even if no exception is thrown: see https://github.com/pytorch/pytorch/pull/125790.


## Toolbox Usage (WIP)

> NOTE:
> 1. We support multi-nodes distributed training based on SLURM. If your resource platform is different and requires multi-nodes distributed training, you may need adapt the shell scripts to your platform. Or you can mult the node count to gradient_accumulation_steps option. Concat us in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) for more support.
> 2. The v2 in script name to record data version is response format version, not dataset version. `dota1-v2` means DOTA-v1.0 of 2-th response.
> 3. The users may misunderstand the data split name. We use `trainval` to represent all the default training split (training with `trainval` if `val` exist, else `train` only. testing with `test` only). However, as is described in the paper, the mF1 calculation requires ground-truth for evaluation. Hence, we add `-train` behind the dataset name to indicate only using `train` for training and `val` for evaluation. (Contact me in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) if there are still confusing things. I paint a pie to refactor this in future.)

### Practices

- train an aerial detector based on [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) on DOTA-v1.0:
```shell
srun ... bash scripts/florence-2-l_vis1024-lang2048_dota1-v2_b2x16-100e.sh
bash scripts/florence-2-l_vis1024-lang2048_dota1-v2_b2x8xga2-100e.sh
```

- evaluate the model on DOTA-v1.0:
```shell
srun ... bash scripts/eval_slurm.sh <checkpoint path>
bash scripts/eval_standalone.sh <checkpoint path>
```

- visualization (for sampled 20 figures)
```shell
bash scripts/eval_standalone.sh <checkpoint path> --shuffle_seed 42 --clip_num 20
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
- ...... (Concat me in [issue](https://github.com/Li-Qingyun/mllm-mmrotate/issues) if there are questions.)

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
