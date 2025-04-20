# Spandrel
Spandrel gives your project support for various PyTorch architectures meant for
AI Super-Resolution, restoration, and inpainting. 

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- einops
- safetensors

`trt` requires additional packages:
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0.dev or later

To install the latest nightly build of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/test/cu128
pip install --no-deps -U torch_tensorrt --index-url https://download.pytorch.org/whl/test/cu128
pip install -U tensorrt-cu12 tensorrt-cu12_bindings tensorrt-cu12_libs --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsspandrel
```

## Usage
```python
from vsspandrel import vsspandrel

ret = vsspandrel(clip, model_path="") # model path is required!
```

See `__init__.py` for the description of the parameters.
