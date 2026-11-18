"""Express a fixed ONNX log-sum-exp with an explicit maximum shift."""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def lower(source, output):
    source, output = Path(source), Path(output)
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError('Expected a new derivative ONNX path')
    model = onnx.load(source)
    graph = model.graph
    originals = [node for node in graph.node if node.op_type == 'ReduceLogSumExp']
    if len(originals) != 1:
        raise ValueError('Expected exactly one mixture log-sum-exp')
    original, = originals
    attributes = {a.name: helper.get_attribute_value(a) for a in original.attribute}
    if (set(attributes) - {'axes', 'keepdims'} or attributes.get('axes') != [1]
            or attributes.get('keepdims', 1) not in (0, 1) or len(original.input) != 1):
        raise ValueError('Expected the fixed mixture component axis')
    prefix = 'stable_mixture_lse_'
    if any(name.startswith(prefix) for node in graph.node for name in node.output):
        raise ValueError('Stable expansion is already present')
    maximum, shifted, exponential, summed, logarithm, total = [prefix + name for name in
                                                              ('max', 'shifted', 'exp', 'sum', 'log', 'total')]
    axis = prefix + 'axis'
    graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axis))
    nodes = [helper.make_node('ReduceMax', [original.input[0]], [maximum], axes=[1], keepdims=1),
             helper.make_node('Sub', [original.input[0], maximum], [shifted]),
             helper.make_node('Exp', [shifted], [exponential]),
             helper.make_node('ReduceSum', [exponential, axis], [summed], keepdims=1),
             helper.make_node('Log', [summed], [logarithm]),
             helper.make_node('Add', [logarithm, maximum], [total])]
    if attributes.get('keepdims', 1):
        nodes.append(helper.make_node('Identity', [total], list(original.output)))
    else:
        nodes.append(helper.make_node('Squeeze', [total, axis], list(original.output)))
    rewritten = []
    for node in graph.node:
        rewritten.extend(nodes if node is original else [node])
    del graph.node[:]
    graph.node.extend(rewritten)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return dict(transformation='max + log(sum(exp(component - max)))', component_axis=1,
                parameters='Original trained weights, unchanged', semantics='Equivalent for finite component values')
