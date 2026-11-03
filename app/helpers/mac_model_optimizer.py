"""
Prepares ONNX models for fast inference on Apple Silicon.

The stock models are tuned for NVIDIA GPUs: many are stored as fp16 graphs full of
Cast nodes, which onnxruntime's CPU and CoreML providers run very slowly (or reject).
For every model we write an optimized copy next to the original that is
  1. upcast to fp32 (CoreML still computes in fp16 on the GPU / Neural Engine),
  2. shape-inferred and constant-folded with onnxsim, so CoreML can take whole graphs.
The optimized copies are cached in model_assets/mac_optimized and reused afterwards.
"""
import os
import threading

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

_lock = threading.Lock()


def _upcast_fp16(model: onnx.ModelProto) -> bool:
    changed = False
    graph_list = [model.graph]
    while graph_list:
        graph = graph_list.pop()
        for idx, init in enumerate(graph.initializer):
            if init.data_type == TensorProto.FLOAT16:
                arr = numpy_helper.to_array(init).astype(np.float32)
                graph.initializer[idx].CopyFrom(numpy_helper.from_array(arr, init.name))
                changed = True
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                vi.type.tensor_type.elem_type = TensorProto.FLOAT
                changed = True
        for node in graph.node:
            for attr in node.attribute:
                if node.op_type == 'Cast' and attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    changed = True
                elif attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
                    changed = True
                elif attr.type == onnx.AttributeProto.GRAPH:
                    graph_list.append(attr.g)
    return changed


def _remove_identity_casts(model: onnx.ModelProto):
    """Turn Cast(float->float) nodes left over after upcasting into Identity, which onnxruntime
    folds away. Rewiring consumers instead breaks graphs whose subgraphs reference the cast."""
    graph = model.graph
    elem_types = {vi.name: vi.type.tensor_type.elem_type for vi in list(graph.input) + list(graph.value_info)}
    for init in graph.initializer:
        elem_types[init.name] = init.data_type
    for node in graph.node:
        if (node.op_type == 'Cast' and node.attribute[0].i == TensorProto.FLOAT
                and elem_types.get(node.input[0]) == TensorProto.FLOAT):
            node.op_type = 'Identity'
            del node.attribute[:]


def _remove_broadcast_expands(model: onnx.ModelProto):
    """Drop Expand nodes that only feed elementwise ops whose other operand already has the
    expanded shape. Plain broadcasting gives the same result and CoreML crashes on some of
    these Expand+Mul patterns (e.g. GhostFace AAD blocks)."""
    graph = model.graph
    try:
        inferred = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        return
    shapes = {}
    for vi in list(inferred.input) + list(inferred.value_info) + list(inferred.output):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField('dim_value') for d in dims):
            shapes[vi.name] = [d.dim_value for d in dims]
    consumers = {}
    for node in graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    output_names = {o.name for o in graph.output}
    elementwise = {'Mul', 'Add', 'Sub', 'Div'}
    remove = set()
    for node in graph.node:
        if node.op_type != 'Expand':
            continue
        out = node.output[0]
        out_shape = shapes.get(out)
        users = consumers.get(out, [])
        if out_shape is None or out in output_names or not users:
            continue
        safe = True
        for user in users:
            others = [i for i in user.input if i != out]
            if user.op_type not in elementwise or len(others) != 1 or shapes.get(others[0]) != out_shape:
                safe = False
                break
        if not safe:
            continue
        for user in users:
            user.input[:] = [node.input[0] if i == out else i for i in user.input]
        remove.add(id(node))
    if remove:
        keep = [n for n in graph.node if id(n) not in remove]
        del graph.node[:]
        graph.node.extend(keep)


def _decompose_instance_norm(model: onnx.ModelProto):
    """Rewrite 4D InstanceNormalization as ReduceMean/Sub/Mul/Sqrt. CoreML's native
    instance_norm fails at runtime on several swapper graphs (GhostFace), the
    decomposed form runs fine on the GPU / Neural Engine."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    opset = max((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), default=13)
    new_nodes = []
    changed = False
    for node in graph.node:
        if node.op_type != 'InstanceNormalization' or node.input[1] not in inits or node.input[2] not in inits:
            new_nodes.append(node)
            continue
        eps = next((a.f for a in node.attribute if a.name == 'epsilon'), 1e-5)
        x, out, p = node.input[0], node.output[0], node.output[0] + '_in_'
        scale = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32).reshape(1, -1, 1, 1)
        bias = numpy_helper.to_array(inits[node.input[2]]).astype(np.float32).reshape(1, -1, 1, 1)
        graph.initializer.extend([numpy_helper.from_array(scale, p + 'scale'),
                                  numpy_helper.from_array(bias, p + 'bias'),
                                  numpy_helper.from_array(np.array(eps, dtype=np.float32), p + 'eps')])
        if opset >= 18:
            graph.initializer.append(numpy_helper.from_array(np.array([2, 3], dtype=np.int64), p + 'axes'))
            reduce = lambda i, o: onnx.helper.make_node('ReduceMean', [i, p + 'axes'], [o], keepdims=1)
        else:
            reduce = lambda i, o: onnx.helper.make_node('ReduceMean', [i], [o], axes=[2, 3], keepdims=1)
        new_nodes += [
            reduce(x, p + 'mean'),
            onnx.helper.make_node('Sub', [x, p + 'mean'], [p + 'd']),
            onnx.helper.make_node('Mul', [p + 'd', p + 'd'], [p + 'd2']),
            reduce(p + 'd2', p + 'var'),
            onnx.helper.make_node('Add', [p + 'var', p + 'eps'], [p + 've']),
            onnx.helper.make_node('Sqrt', [p + 've'], [p + 'std']),
            onnx.helper.make_node('Div', [p + 'd', p + 'std'], [p + 'n']),
            onnx.helper.make_node('Mul', [p + 'n', p + 'scale'], [p + 'ns']),
            onnx.helper.make_node('Add', [p + 'ns', p + 'bias'], [out]),
        ]
        changed = True
    if changed:
        del graph.node[:]
        graph.node.extend(new_nodes)


def _decompose_reflect_pad(model: onnx.ModelProto):
    """Rewrite spatial reflect-mode Pad as Slice+Concat. CoreML EP doesn't take reflect Pad,
    so each one splits the graph into a new CoreML partition (inswapper ends up with 14)."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_nodes = []
    changed = False
    for node in graph.node:
        mode = next((a.s for a in node.attribute if a.name == 'mode'), b'constant')
        if node.op_type != 'Pad' or mode != b'reflect' or len(node.input) < 2 or node.input[1] not in inits:
            new_nodes.append(node)
            continue
        pads = numpy_helper.to_array(inits[node.input[1]]).tolist()
        if len(pads) != 8 or any(pads[i] for i in (0, 1, 4, 5)):
            new_nodes.append(node)
            continue
        cur, p = node.input[0], node.output[0] + '_rp_'
        for axis, (before, after) in ((2, (pads[2], pads[6])), (3, (pads[3], pads[7]))):
            if before == 0 and after == 0:
                continue
            parts = []
            for side, n in (('b', before), ('a', after)):
                if n == 0:
                    continue
                # before: rows n..1, after: rows -2..-(n+1)  (both reversed)
                starts, ends = ([n], [0]) if side == 'b' else ([-2], [-(n + 2)])
                key = f'{p}{axis}{side}'
                graph.initializer.extend([
                    numpy_helper.from_array(np.array(starts, dtype=np.int64), key + '_s'),
                    numpy_helper.from_array(np.array(ends, dtype=np.int64), key + '_e'),
                    numpy_helper.from_array(np.array([axis], dtype=np.int64), key + '_ax'),
                    numpy_helper.from_array(np.array([-1], dtype=np.int64), key + '_st'),
                ])
                new_nodes.append(onnx.helper.make_node('Slice', [cur, key + '_s', key + '_e', key + '_ax', key + '_st'], [key]))
                parts.append((side, key))
            ordered = [k for s, k in parts if s == 'b'] + [cur] + [k for s, k in parts if s == 'a']
            out = f'{p}{axis}cat'
            new_nodes.append(onnx.helper.make_node('Concat', ordered, [out], axis=axis))
            cur = out
        new_nodes.append(onnx.helper.make_node('Identity', [cur], [node.output[0]]))
        changed = True
    if changed:
        del graph.node[:]
        graph.node.extend(new_nodes)


def optimized_model_path(model_path: str) -> str:
    """Return the path of an Apple-Silicon friendly copy of model_path, creating it if needed.
    Falls back to the original path if anything goes wrong."""
    model_path = os.path.abspath(model_path)
    out_dir = os.path.join(os.path.dirname(model_path), 'mac_optimized')
    out_path = os.path.join(out_dir, os.path.basename(model_path).replace('.fp16', ''))
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(model_path):
        return out_path
    with _lock:
        if os.path.exists(out_path):
            return out_path
        try:
            os.makedirs(out_dir, exist_ok=True)
            # Models above 2GB (external data) are left as-is
            if os.path.getsize(model_path) > 1_900_000_000:
                return model_path
            model = onnx.load(model_path)
            if _upcast_fp16(model):
                try:
                    model = onnx.shape_inference.infer_shapes(model)
                except Exception:
                    pass
                _remove_identity_casts(model)
            try:
                import onnxsim
                model_sim, ok = onnxsim.simplify(model)
                if ok:
                    model = model_sim
            except Exception as e:
                print(f"[mac-optimizer] onnxsim skipped for {os.path.basename(model_path)}: {e}")
            _remove_broadcast_expands(model)
            _decompose_instance_norm(model)
            _decompose_reflect_pad(model)
            del model.graph.value_info[:]
            tmp_path = out_path + '.tmp'
            onnx.save(model, tmp_path)
            # make sure the rewritten graph still loads before using it
            import onnxruntime
            check_opts = onnxruntime.SessionOptions()
            check_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            onnxruntime.InferenceSession(tmp_path, sess_options=check_opts, providers=['CPUExecutionProvider'])
            os.replace(tmp_path, out_path)
            print(f"[mac-optimizer] Optimized {os.path.basename(model_path)}")
            return out_path
        except Exception as e:
            print(f"[mac-optimizer] Could not optimize {os.path.basename(model_path)}, using original: {e}")
            if os.path.exists(out_path + '.tmp'):
                os.remove(out_path + '.tmp')
            return model_path
