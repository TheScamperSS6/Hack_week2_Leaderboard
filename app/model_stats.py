from math import prod
from pathlib import Path
from typing import Any


FLOPS_PER_MAC = 2


def estimate_onnx_gflops(model_path: str | Path) -> float:
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as exc:
        raise RuntimeError("onnx is required to estimate model GFLOPs") from exc

    model = onnx.load(str(model_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    initializers = {
        initializer.name: tuple(int(dim) for dim in initializer.dims)
        for initializer in model.graph.initializer
    }
    shapes = collect_tensor_shapes(model)

    flops = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            flops += conv_flops(node, shapes, initializers)
        elif node.op_type == "Gemm":
            flops += gemm_flops(node, shapes, initializers)
        elif node.op_type == "MatMul":
            flops += matmul_flops(node, shapes, initializers)

    if flops <= 0:
        raise RuntimeError(f"could not infer GFLOPs from ONNX model: {model_path}")

    return round(flops / 1_000_000_000, 6)


def collect_tensor_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    value_infos = [
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    ]
    for value_info in value_infos:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue

        dims: list[int] = []
        complete = True
        for index, dim in enumerate(tensor_type.shape.dim):
            if dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif index == 0:
                dims.append(1)
            else:
                complete = False
                break

        if complete and dims:
            shapes[value_info.name] = tuple(dims)

    for initializer in model.graph.initializer:
        shapes[initializer.name] = tuple(int(dim) for dim in initializer.dims)

    return shapes


def conv_flops(node: Any, shapes: dict[str, tuple[int, ...]], initializers: dict[str, tuple[int, ...]]) -> int:
    if len(node.input) < 2 or not node.output:
        return 0

    weight_shape = initializers.get(node.input[1]) or shapes.get(node.input[1])
    output_shape = shapes.get(node.output[0])
    if not weight_shape or not output_shape or len(weight_shape) < 4 or len(output_shape) < 4:
        return 0

    group = int(attribute_value(node, "group", 1))
    out_elements = prod(output_shape)
    kernel_ops = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])
    if group > 1:
        kernel_ops = int(weight_shape[1]) * int(weight_shape[2]) * int(weight_shape[3])

    return int(out_elements * kernel_ops * FLOPS_PER_MAC)


def gemm_flops(node: Any, shapes: dict[str, tuple[int, ...]], initializers: dict[str, tuple[int, ...]]) -> int:
    if len(node.input) < 2 or not node.output:
        return 0

    a_shape = shapes.get(node.input[0])
    b_shape = initializers.get(node.input[1]) or shapes.get(node.input[1])
    output_shape = shapes.get(node.output[0])
    if not a_shape or not b_shape or not output_shape or len(a_shape) < 2 or len(b_shape) < 2:
        return 0

    trans_a = int(attribute_value(node, "transA", 0))
    trans_b = int(attribute_value(node, "transB", 0))
    k = a_shape[-2] if trans_a else a_shape[-1]
    b_k = b_shape[-1] if trans_b else b_shape[-2]
    if k != b_k:
        k = min(int(k), int(b_k))

    return int(prod(output_shape) * int(k) * FLOPS_PER_MAC)


def matmul_flops(node: Any, shapes: dict[str, tuple[int, ...]], initializers: dict[str, tuple[int, ...]]) -> int:
    if len(node.input) < 2 or not node.output:
        return 0

    a_shape = shapes.get(node.input[0])
    b_shape = initializers.get(node.input[1]) or shapes.get(node.input[1])
    output_shape = shapes.get(node.output[0])
    if not a_shape or not b_shape or not output_shape or len(a_shape) < 2 or len(b_shape) < 2:
        return 0

    k = min(int(a_shape[-1]), int(b_shape[-2]))
    return int(prod(output_shape) * k * FLOPS_PER_MAC)


def attribute_value(node: Any, name: str, default: Any) -> Any:
    for attribute in node.attribute:
        if attribute.name != name:
            continue
        if attribute.type == 2:
            return attribute.i
        if attribute.type == 1:
            return attribute.f
        if attribute.type == 7:
            return list(attribute.ints)
    return default
