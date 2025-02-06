from __future__ import annotations

import math
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .spandrel.libs.spandrel.spandrel import ModelLoader


__version__ = "1.1.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")



class Backend:
    @dataclass
    class Torch:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


@redirect_stdout_to_stderr()
@torch.inference_mode()
def vsspandrel(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    batch_size: int = 1,
    model_path: str | None = None,
    denoise_strength: float = 0.5,
    tile: list[int] = [0, 0],
    tile_pad: int = 8,
    trt: bool = False,
    trt_static_shape: bool = True,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [720, 480],
    trt_max_shape: list[int] = [1920, 1080],
    trt_debug: bool = False,
    trt_mixed_precision: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str | None = None,
) -> vs.VideoNode:
    """Training Real-World Blind Super-Resolution with Pure Synthetic Data

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param batch_size:              Number of frames per batch.
    :param model:                   Model to use. Ignored if model_path is specified.
    :param auto_download:           Automatically download the specified model if the file has not been downloaded.
    :param model_path:              Path to custom model file.
    :param denoise_strength:        Denoise strength for realesr-general-x4v3 model.
                                    0 for weak denoise (keep noise), 1 for strong denoise ability.
    :param tile:                    Tile width and height. As too large images result in the out of GPU memory issue, so
                                    this tile option will first crop input images into tiles, and then process each of
                                    them. Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_static_shape:        Build with static or dynamic shapes.
    :param trt_min_shape:           Min size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_opt_shape:           Opt size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_max_shape:           Max size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_mixed_precision:     Use multiple precisions for TensorRT engine, used for models with DySample
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("spandrel: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("spandrel: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("spandrel: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("spandrel: num_streams must be at least 1")

    if batch_size < 1:
        raise vs.Error("spandrel: batch_size must be at least 1")

    if denoise_strength < 0 or denoise_strength > 1:
        raise vs.Error("spandrel: denoise_strength must be between 0.0 and 1.0 (inclusive)")

    if not isinstance(tile, list) or len(tile) != 2:
        raise vs.Error("spandrel: tile must be a list with 2 items")
    
    if model_path is None:
        raise vs.Error("spandrel: model_path must be specified")

    if not trt_static_shape:
        if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
            raise vs.Error("spandrel: trt_min_shape must be a list with 2 items")

        if any(trt_min_shape[i] < 1 for i in range(2)):
            raise vs.Error("spandrel: trt_min_shape must be at least 1")

        if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
            raise vs.Error("spandrel: trt_opt_shape must be a list with 2 items")

        if any(trt_opt_shape[i] < 1 for i in range(2)):
            raise vs.Error("spandrel: trt_opt_shape must be at least 1")

        if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
            raise vs.Error("spandrel: trt_max_shape must be a list with 2 items")

        if any(trt_max_shape[i] < 1 for i in range(2)):
            raise vs.Error("spandrel: trt_max_shape must be at least 1")

        if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
            raise vs.Error("spandrel: trt_min_shape must be less than trt_max_shape")
    
    

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)
    
    model_path = os.path.realpath(model_path)
    model_name = os.path.basename(model_path)

    if trt_cache_dir is None:
        trt_cache_dir = os.path.dirname(os.path.abspath(model_path))

    module = ModelLoader().load_from_file(model_path)

    model = module.model
    scale = module.scale

    model.eval().to(device, dtype)

    match scale:
        case 1:
            modulo = 4
        case 2:
            modulo = 2
        case _:
            modulo = 1

    if all(t > 0 for t in tile):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, clip.width) / modulo) * modulo
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, clip.height) / modulo) * modulo
    else:
        pad_w = math.ceil(clip.width / modulo) * modulo
        pad_h = math.ceil(clip.height / modulo) * modulo

    if trt:
        import tensorrt
        import torch_tensorrt

        if trt_static_shape:
            dimensions = f"{pad_w}x{pad_h}"
        else:
            for i in range(2):
                trt_min_shape[i] = math.ceil(trt_min_shape[i] / modulo) * modulo
                trt_opt_shape[i] = math.ceil(trt_opt_shape[i] / modulo) * modulo
                trt_max_shape[i] = math.ceil(trt_max_shape[i] / modulo) * modulo

            dimensions = (
                f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
                f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
                f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
            )

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_batch-{batch_size}"
                + f"_{dimensions}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ("_mixed_precision" if trt_mixed_precision else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")

            example_inputs = (torch.zeros([batch_size, 3, pad_h, pad_w], dtype=dtype, device=device),)

            if trt_static_shape:
                dynamic_shapes = None

                inputs = example_inputs
            else:
                trt_min_shape.reverse()
                trt_opt_shape.reverse()
                trt_max_shape.reverse()

                _height = torch.export.Dim("height", min=trt_min_shape[0] // modulo, max=trt_max_shape[0] // modulo)
                _width = torch.export.Dim("width", min=trt_min_shape[1] // modulo, max=trt_max_shape[1] // modulo)
                dim_height = _height * modulo
                dim_width = _width * modulo
                dynamic_shapes = {"x": {2: dim_height, 3: dim_width}}

                inputs = [
                    torch_tensorrt.Input(
                        min_shape=[batch_size, 3] + trt_min_shape,
                        opt_shape=[batch_size, 3] + trt_opt_shape,
                        max_shape=[batch_size, 3] + trt_max_shape,
                        dtype=dtype,
                        name="x",
                    )
                ]
            
            # workaround to get update_params to work without tensorrt getting messed up
            model1 = model
            model1(inputs[0])
            model.load_state_dict(model.state_dict())
            del model1
            torch.cuda.empty_cache()

            
            from torch_tensorrt.dynamo.conversion.impl.grid import GridSamplerInterpolationMode
            from torch._export.converter import TS2EPConverter
            from torch.export.exported_program import ExportedProgram
            
            def torchscript_to_dynamo(
                model: torch.nn.Module, example_inputs: tuple[torch.Tensor]
            ) -> ExportedProgram:
                """Converts a TorchScript module to a Dynamo program."""
                module = torch.jit.trace(model, example_inputs)
                exported_program = TS2EPConverter(
                    module, sample_args=tuple(example_inputs), sample_kwargs=None
                ).convert()
                del module
                torch.cuda.empty_cache()
                return exported_program

            def nnmodule_to_dynamo(
                model: torch.nn.Module, example_inputs: tuple[torch.Tensor], dynamic_shapes=None
                ) -> ExportedProgram:
                    """Converts a nn.Module to a Dynamo program."""
                    return torch.export.export(
                        model, tuple(example_inputs), dynamic_shapes=dynamic_shapes
                )

            try:
                exported_program = nnmodule_to_dynamo(model, example_inputs, dynamic_shapes=dynamic_shapes)
            except Exception:
                exported_program = torchscript_to_dynamo(model, example_inputs) # Fallback to torchscript if nnmodule_to_dynamo fails, should be dynamic but i cant find any docs on the function

            

            GridSamplerInterpolationMode.update(
                {
                    0: tensorrt.InterpolationMode.LINEAR,
                    1: tensorrt.InterpolationMode.NEAREST,
                    2: tensorrt.InterpolationMode.CUBIC,
                }
            )
            
            if trt_mixed_precision:
    
                module = torch_tensorrt.dynamo.compile(
                    exported_program,
                    inputs,
                    device=device,
                    use_explicit_typing=True,
                    debug=trt_debug,
                    num_avg_timing_iters=4,
                    workspace_size=trt_workspace_size,
                    min_block_size=1,
                    max_aux_streams=trt_max_aux_streams,
                    optimization_level=trt_optimization_level,
                )
            else:
                
                module = torch_tensorrt.dynamo.compile(
                    exported_program,
                    inputs,
                    device=device,
                    enabled_precisions={dtype},
                    debug=trt_debug,
                    num_avg_timing_iters=4,
                    workspace_size=trt_workspace_size,
                    min_block_size=1,
                    max_aux_streams=trt_max_aux_streams,
                    optimization_level=trt_optimization_level,
                )

            torch_tensorrt.save(module, trt_engine_path, output_format="torchscript", inputs=example_inputs)

        module = [torch.jit.load(trt_engine_path).eval() for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        backend = Backend.Torch(model)

    index = -1
    index_lock = Lock()

    inf_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    f2t_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    t2f_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    inf_stream_locks = [Lock() for _ in range(num_streams)]
    f2t_stream_locks = [Lock() for _ in range(num_streams)]
    t2f_stream_locks = [Lock() for _ in range(num_streams)]

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with f2t_stream_locks[local_index], torch.cuda.stream(f2t_streams[local_index]):
            img = torch.stack([frame_to_tensor(f[i], device) for i in range(batch_size)])

            f2t_streams[local_index].synchronize()

        with inf_stream_locks[local_index], torch.cuda.stream(inf_streams[local_index]):
            if all(t > 0 for t in tile):
                output = tile_process(img, scale, tile, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                if need_pad := pad_w - w > 0 or pad_h - h > 0:
                    img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "replicate")

                if trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                if need_pad:
                    output = output[:, :, : h * scale, : w * scale]

            inf_streams[local_index].synchronize()

        with t2f_stream_locks[local_index], torch.cuda.stream(t2f_streams[local_index]):
            frame = tensor_to_frame(output[0], f[batch_size].copy(), t2f_streams[local_index])
            for i in range(1, batch_size):
                frame.props[f"vsspandrel_batch_frame{i}"] = tensor_to_frame(
                    output[i], f[batch_size].copy(), t2f_streams[local_index]
                )
            return frame

    if (pad := (batch_size - clip.num_frames % batch_size) % batch_size) > 0:
        clip = clip.std.DuplicateFrames([clip.num_frames - 1] * pad)

    clips = [clip[i::batch_size] for i in range(batch_size)]
    new_clip = clips[0].std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    new_clip = new_clip.std.CopyFrameProps(clips[0])
    clips.append(new_clip)

    outputs = [new_clip.std.FrameEval(lambda n: new_clip.std.ModifyFrame(clips, inference), clip_src=clips)]
    for i in range(1, batch_size):
        outputs.append(outputs[0].std.PropToClip(f"vsspandrel_batch_frame{i}"))

    output = vs.core.std.Interleave(outputs)
    if pad > 0:
        output = output[:-pad]
    return output


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Torch | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            if need_pad := pad_w - w > 0 or pad_h - h > 0:
                input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), "replicate")

            # process tile
            if isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            if need_pad:
                output_tile = output_tile[:, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
