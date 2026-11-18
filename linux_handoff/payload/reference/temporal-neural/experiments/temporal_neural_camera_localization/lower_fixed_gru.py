"""Lower this pilot's fixed forward GRU into equivalent ONNX arithmetic.

Weights are reused exactly; no training or dataset mutation is performed.
The narrow contract is one forward GRU, zero initial state, batch one,
linear_before_reset=1, default Sigmoid/Tanh, and four or eight frames.
Reference: https://onnx.ai/onnx/operators/onnx__GRU.html
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def lower(source, output, window):
    source, output = Path(source), Path(output)
    if window not in (4, 8) or output.exists() or source.resolve() == output.resolve():
        raise ValueError('Expected a new derivative path and a four/eight-frame window')
    model = onnx.load(source)
    graph = model.graph
    nodes = [node for node in graph.node if node.op_type == 'GRU']
    if len(nodes) != 1 or any(name.startswith('fixed_gru_') for node in graph.node for name in node.output):
        raise ValueError('Expected exactly one original GRU')
    gru, = nodes
    attributes = {a.name: helper.get_attribute_value(a) for a in gru.attribute}
    allowed = {'hidden_size', 'linear_before_reset', 'direction', 'layout', 'activations'}
    if (set(attributes) - allowed or attributes.get('linear_before_reset', 0) != 1
            or attributes.get('direction', b'forward') != b'forward'
            or attributes.get('layout', 0) != 0
            or attributes.get('activations', [b'Sigmoid', b'Tanh']) != [b'Sigmoid', b'Tanh']):
        raise ValueError('Unsupported GRU attributes')
    if len(gru.input) != 6 or gru.input[4] or len(gru.output) != 2:
        raise ValueError('Expected full fixed-length sequences and an explicit zero initial state')
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    if input_shape != [1, window * 3, 96, 160]:
        raise ValueError('Expected this pilot fixed packed RGB shape')
    values = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    initial = producers.get(gru.input[5])
    if initial is not None and initial.op_type == 'Expand':
        initial = producers.get(initial.input[0])
    if initial is None or initial.op_type != 'Constant':
        raise ValueError('Initial state must be an explicit zero constant')
    initial_value = numpy_helper.to_array(next(a.t for a in initial.attribute if a.name == 'value'))
    hidden = attributes['hidden_size']
    if initial_value.shape != (1, 1, hidden) or np.any(initial_value != 0):
        raise ValueError('Only the original zero initial state is supported')
    w, r, bias = (values[name] for name in gru.input[1:4])
    if w.shape != (1, 3 * hidden, hidden) or r.shape != w.shape or bias.shape != (1, 6 * hidden):
        raise ValueError('Unexpected pilot GRU weight dimensions')

    replacement, added = [], []
    counter = 0

    def constant(label, array):
        name = 'fixed_gru_' + label
        added.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def op(kind, inputs, **kwargs):
        nonlocal counter
        counter += 1
        name = 'fixed_gru_value_' + str(counter)
        replacement.append(helper.make_node(kind, inputs, [name], name=name, **kwargs))
        return name

    w_name = constant('W_transpose', w[0].T.copy())
    r_name = constant('R_transpose', r[0].T.copy())
    wb = constant('W_bias', bias[0, :3 * hidden].copy())
    rb = constant('R_bias', bias[0, 3 * hidden:].copy())
    h = constant('initial_h', np.zeros((1, hidden), dtype=w.dtype))
    one = constant('one', np.ones((), dtype=w.dtype))
    axis = constant('slice_axis', np.array([1], dtype=np.int64))
    axes_y = constant('sequence_axes', np.array([0, 1], dtype=np.int64))
    axes_h = constant('state_axis', np.array([0], dtype=np.int64))
    bounds = [(constant('start_' + str(i), np.array([i * hidden], dtype=np.int64)),
               constant('end_' + str(i), np.array([(i + 1) * hidden], dtype=np.int64)))
              for i in range(3)]
    history = []
    for step in range(window):
        index = constant('step_' + str(step), np.array(step, dtype=np.int64))
        x = op('Gather', [gru.input[0], index], axis=0)
        xg = op('Add', [op('MatMul', [x, w_name]), wb])
        hg = op('Add', [op('MatMul', [h, r_name]), rb])
        xz, xr, xn = [op('Slice', [xg, start, end, axis]) for start, end in bounds]
        hz, hr, hn = [op('Slice', [hg, start, end, axis]) for start, end in bounds]
        z = op('Sigmoid', [op('Add', [xz, hz])])
        reset = op('Sigmoid', [op('Add', [xr, hr])])
        candidate = op('Tanh', [op('Add', [xn, op('Mul', [reset, hn])])])
        h = op('Add', [op('Mul', [op('Sub', [one, z]), candidate]), op('Mul', [z, h])])
        history.append(op('Unsqueeze', [h, axes_y]))
    replacement.extend([
        helper.make_node('Concat', history, [gru.output[0]], axis=0),
        helper.make_node('Unsqueeze', [h, axes_h], [gru.output[1]])])
    rewritten = []
    for node in graph.node:
        rewritten.extend(replacement if node is gru else [node])
    # Drop the original now-unused dynamic-state machinery and GRU weights.
    needed = {value.name for value in graph.output}
    kept = []
    for node in reversed(rewritten):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    all_initializers = list(graph.initializer) + added
    del graph.initializer[:]
    graph.initializer.extend(item for item in all_initializers if item.name in needed)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    if any(node.op_type == 'GRU' for node in model.graph.node):
        raise ValueError('GRU lowering did not replace the original node')
    onnx.save(model, output)
    return dict(window=window, hidden=hidden, linear_before_reset=1,
                weights='Original ONNX weights, transposed/sliced without retraining',
                reference='https://onnx.ai/onnx/operators/onnx__GRU.html',
                operators=sorted({node.op_type for node in model.graph.node}))
