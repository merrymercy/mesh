import numpy as np
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import mesh_tensorflow.auto_mtf
from mesh_tensorflow.auto_mtf import memory_estimator, layout_optimizer


def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def solve_layout(mtf_graph, mesh_shape, mtf_outputs=()):
  mesh_shape = mtf.convert_to_shape(mesh_shape)
  estimator = memory_estimator.MemoryEstimator(mtf_graph, mesh_shape,
                                               mtf_outputs)
  optimizer = layout_optimizer.LayoutOptimizer(estimator)
  ret = optimizer.solve()
  return mtf.convert_to_layout_rules(ret)


ct = 0
def dim(name, size):
    global ct
    ct += 1
    name = f"{name}_{ct}"
    return mtf.Dimension(name, size)


def benchmark_mlp_one_case(case):
    batch_size, seq_len, hidden_size, num_layers, dp_size, tensor_mp_size = case
    batch_size = batch_size * seq_len

    # Create graph and mesh
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
    mesh_shape = [(f"0len_{dp_size}", dp_size), (f"1len_{tensor_mp_size}", tensor_mp_size)]

    # Model definition
    data = mtf.get_variable(mesh, "data",
                            [dim("batch", batch_size), dim("hidden0", hidden_size)],
                            initializer=tf.ones)

    net = data
    for i in range(1, num_layers + 1):
        net = mtf.layers.dense(net, dim("hidden_x4", hidden_size * 4),
                               reduced_dims=net.shape.dims[-1:])
        net = mtf.layers.dense(net, dim("hidden1", hidden_size),
                               reduced_dims=net.shape.dims[-1:])
    label = mtf.get_variable(mesh, "label", net.shape, initializer=tf.ones)

    loss = mtf.reduce_mean(mtf.square(net - label))
    layout_rules = solve_layout(graph, mesh_shape, [loss])
    print(layout_rules)

    # Backward
    trainable_variables = graph.trainable_variables[2:]
    var_grads = mtf.gradients([loss], [v.outputs[0] for v in trainable_variables])
    optimizer = mtf.optimize.SgdOptimizer(1e-2)
    update_ops = optimizer.apply_grads(var_grads, trainable_variables)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
                mesh_shape, layout_rules, devices)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    tf_update_ops = [lowering.lowered_operation(x) for x in update_ops]

    # Init
    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(tf_group)

    # Benchmark
    warmup = 2
    repeat = 2
    number = 5

    for i in range(warmup):
        session.run(tf_update_ops)

    costs = []
    for i in range(repeat):
        tic = time.time()
        for j in range(number):
            session.run(tf_update_ops)
        toc = time.time()
        costs.append((toc - tic) / number)

    heads = ["Case", "Mean Time", "Std Time"]
    values = [str(case), f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, "result_mlp.tsv")
    print("")


benchmark_suite = [
    # Batch size, seq_len, hidden size, num_layers, dp_size, tensor_mp_size,
    #(16,          1024,    2304,        4,          4,       1),
    #(16,          1024,    2304,        4,          2,       2),
    #(16,          1024,    2304,        4,          1,       4),

    ## Batch size, seq_len, hidden size, num_layers, dp_size, tensor_mp_size,
    #(8,           256,     5760,        4,          4,       1),
    #(8,           256,     5760,        4,          2,       2),
    #(8,           256,     5760,        4,          1,       4),

    (32,           1024,    2304,        4,          4,       1),
    (32,           1024,    2304,        4,          2,       2),
    (8,            256,     5760,        4,          1,       4),
    (8,            256,     5760,        4,          2,       2),
]


def benchmark_mlp():
    for i, case in enumerate(benchmark_suite):
        with tf.variable_scope(f"case_{i}"):
            benchmark_mlp_one_case(case)


if __name__ == "__main__":
    tf.disable_v2_behavior()

    benchmark_mlp()

