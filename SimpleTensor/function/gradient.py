# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

from SimpleTensor import runtime, Clip
from SimpleTensor import Node, Operation
import numpy as np

# register module
def __get_grad_by_shape(node : Node, grad : np.ndarray):
        node_shape, grad_shape = node.shape, grad.shape
        if node_shape == grad_shape:
            return grad
        else:
            for axis, _ in enumerate(grad_shape):
                if grad_shape[axis] != node_shape[axis]:
                    break
            return grad.mean(axis=axis).reshape(node_shape)

# TODO : In order to avoid necessary misunderstanding, please notice that averary of loss have been implemented by add
@runtime.gradient_func("add")
def __add_gradient(op_node : Operation, grad : np.ndarray):
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]

@runtime.gradient_func("minus")
def __minus_gradient(op_node : Operation, grad : np.ndarray):
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        -1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]

@runtime.gradient_func("negative")
def __negative_gradient(op_node : Operation, grad : np.ndarray):
    return [-1. * grad]

@runtime.gradient_func("elementwise_pow")
def __elementwise_pow_gradeint(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.y
    return [y * (x ** (y - 1)) * grad]

@runtime.gradient_func("matmul")
def __matmul_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return [grad @ y.T, x.T @ grad]

@runtime.gradient_func("multiply")
def __multiply_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return [y * grad, x * grad]

@runtime.gradient_func("reduce_sum")
def __reduce_sum_gradient(op_node : Operation, grad : np.ndarray):
    return [1. * grad]

@runtime.gradient_func("reduce_mean")
def __reduce_mean_gradient(op_node : Operation, grad : np.ndarray):
    # multiplier = op_node.input_nodes[0].data.size / op_node.data.size
    # multiplier = op_node.input_nodes[0].shape[0]
    multiplier = 1
    return [1. / multiplier * grad]

@runtime.gradient_func("log")
def __log_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    return [1. / x * grad]

@runtime.gradient_func("sigmoid")
def __sigmoid_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    ex = Clip.EXP(x)
    return [ex / ((1 + ex) ** 2) * grad]

@runtime.gradient_func("tanh")
def __tanh_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    e2x = Clip.EXP(2. * x)
    return [4 * e2x / ((1 + e2x) ** 2) * grad]

@runtime.gradient_func("relu")
def __relu_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x
    return [k * grad]

@runtime.gradient_func("leaky_relu")
def __leaky_relu(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x
    return [k * grad]

@runtime.gradient_func("elu")
def __elu_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    k = np.array(x)
    k[k >= 0] = 1.
    k[k < 0] = op_node.alpha * np.exp(k[k < 0])
    return [k * grad]

@runtime.gradient_func("softmax")
def __softmax_gradient(op_node : Operation, grad : np.ndarray):
    f = op_node.data
    return f * (1 - f)