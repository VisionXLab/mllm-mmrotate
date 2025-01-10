# lmmrotate

[ArXiv]() | [ResearchGate (Full Text Available)]() | [Huggingface]()

This repo aims to fine-tune **L**arge **M**ultimodal language **M**odels for rotated object detection as in [MMRotate](https://github.com/open-mmlab/mmrotate) and hosts the official implementation of the paper: **A Simple Aerial Detection Baseline of Multimodal Language Models**, _ArXiv 25xx.xxxxx_, Qingyun Li, Yushi Chen, Xinya Shu, Dong Chen, Xin He, Yi Yu, and Xue Yang.

## Enviroment
**NOTE: a misaligned enviroment between inference and training may cause bad effect.**

- create env and install torch
```
conda create -n lmmrotate python=3.10.12
conda activate lmmrotate
```

- set cuda&gcc (recommanded for current enviroment, you can also set it in ~/.bashrc)
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
touch $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
vim $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh
```
write the following lines
```
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
```
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

- build and install [mmcv](https://mmcv.readthedocs.io/en/latest/)
```
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
```
mim install "mmdet==3.0.0"
mim install "mmrotate==1.0.0rc1"
```

- install [flash-attention](https://github.com/Dao-AILab/flash-attention)
```
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

- install lmmrotate
```
pip install -e .
```

- The torch-2.3.0 may raise a warning as:
> site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.) In version 2.3.0 of pytorch, it prints this unwanted warning even if no exception is thrown: see https://github.com/pytorch/pytorch/pull/125790.
