# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version:1.0.1

import numpy as np
from queue import Queue
from typing import Union, List

class Node : pass
class add : pass
class minus : pass
class negative : pass
class multiply : pass
class matmul : pass
class elementwise_pow : pass

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}
    
    def register(self, target):
        def add_register_items(key, value):
            if not callable(value):
                raise ValueError(f"register object must be callable, but receive {type(value)}")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value
        return add_register_items(target.__name__, target) if callable(target) else lambda x : add_register_items(target, x)

    def __call__(self, *args, **kwargs):
        return self.register(*args, **kwargs)

    def __setitem__(self, key, value):
        self._dict[key] = value
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def __str__(self):
        return f"{str(self._dict)}"

    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

_register_act_functions = Register()
_register_grad_functions = Register()
_default_graph = []

# constant variable
PRECISE_LOW = 1e-127
PRECISE_HIGH = 1e128
EXP_PRECISE_LOW = -292.42
EXP_RPECISE_HIGH = 294.73
EXP = lambda x : np.exp(np.clip(x, EXP_PRECISE_LOW, EXP_RPECISE_HIGH))

class Node(object):
    def __init__(self):
        """
            base for all types of node in the static calculation graph
        """
        self.next_nodes = []
        self.data = None
        _default_graph.append(self)
    
    def __neg__(self):
        return negative(self)

    def __add__(self, node : Node):
        return add(self, node)
    
    def __sub__(self, node : Node):
        return minus(self, node)

    def __mul__(self, node : Node):
        return multiply(self, node)
    
    def __pow__(self, y : Union[int, float]):
        return elementwise_pow(self, y)
    
    def __matmul__(self, node : Node):
        return matmul(self, node)

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.data)})"
    
    @property
    def numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data)

    @property
    def shape(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data).shape
    
    def to_numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data)
    
    def to_list(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return list(self.data)

class Operation(Node):
    def __init__(self, input_nodes : List[Node] = []):
        super().__init__()
        self.input_nodes = input_nodes
        for node in input_nodes:
            node.next_nodes.append(self)
    
    def compute(self, *args):
        pass


class Placeholder(Node):
    def __init__(self):
        super().__init__()

class Variable(Node):
    def __init__(self, init_value : Union[np.ndarray, list] = None):
        super().__init__()
        self.data = init_value

"""
    fundamental function
"""
class add(Operation):
    def __init__(self, x : Node, y : Node):
        super().__init__(input_nodes=[x, y])
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v + y_v

class minus(Operation):
    def __init__(self, x : Node, y : Node):
        super().__init__(input_nodes=[x, y])
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v - y_v

class negative(Operation):
    def __init__(self, x : Node):
        super().__init__(input_nodes=[x])
    
    def compute(self, x_v : np.ndarray):
        return -1. * x_v

class elementwise_pow(Operation):
    def __init__(self, x : Node, y : Union[int, float]):
        super().__init__(input_nodes=[x])
        self.y = y
    
    def compute(self, x_v : np.ndarray):
        return x_v ** self.y

class matmul(Operation):
    def __init__(self, x : Node, y : Node):
        super().__init__(input_nodes=[x, y])
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v @ y_v

class multiply(Operation):
    def __init__(self, x : Node, y : Node):
        super().__init__(input_nodes=[x, y])
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v * y_v

class reduce_sum(Operation):
    def __init__(self, x : Node, axis : int = None):
        super().__init__(input_nodes=[x])
        self.axis = axis
    
    def compute(self, x_v : np.ndarray):
        return np.sum(x_v, axis=self.axis)

class reduce_mean(Operation):
    def __init__(self, x : Node, axis : int = None):
        super().__init__(input_nodes=[x])
        self.axis = axis
    
    def compute(self, x_v : np.ndarray):
        return np.mean(x_v, axis=self.axis)

class log(Operation):
    def __init__(self, x : Node):
        super().__init__(input_nodes=[x])
    
    def compute(self, x_v : np.ndarray):
        return np.log(x_v)

"""
    activate function
"""
@_register_act_functions("sigmoid")
class sigmoid(Operation):
    def __init__(self, x : Node):
        super().__init__(input_nodes=[x])
    
    def compute(self, x_v : np.ndarray):
        return 1 / (1 + EXP(-1. * x_v))

@_register_act_functions("tanh")
class tanh(Operation):
    def __init__(self, x : Node):
        super().__init__(input_nodes=[x])
    
    def compute(self, x_v : np.ndarray):
        # default precise: float32
        ex = EXP(x_v)
        dex = 1. / ex
        return (ex - dex) / (ex + dex)

@_register_act_functions("relu")
class relu(Operation):
    def __init__(self, x : Node):
        super().__init__(input_nodes=[x])
    
    def compute(self, x_v : np.ndarray):
        y = np.array(x_v)
        y[y < 0] = 0.
        return y

@_register_act_functions("leaky_relu")
class leaky_relu(Operation):
    def __init__(self, x : Node, alpha : float = 1e-2):
        super().__init__(input_nodes=[x])
        self.alpha = alpha
    
    def compute(self, x_v : np.ndarray):
        y = np.array(x_v)
        y[y < 0] *= self.alpha
        return y

@_register_act_functions("elu")
class elu(Operation):
    def __init__(self, x : Node, alpha : float = 1e-2):
        super().__init__(input_nodes=[x])
        self.alpha = alpha
    
    def compute(self, x_v : np.ndarray):
        y = np.array(x_v)
        y[y < 0] = self.alpha * (np.exp(y[y < 0]) - 1)
        return y


# TODO : add a class to program the operation which have paramter and can update its self

class softmax(Operation):
    def __init__(self, x : Node, axis : int = None):
        super().__init__(input_nodes=[x])
        self.axis = axis
    
    def compute(self, x_v : np.ndarray):
        ex = np.exp(x_v)
        reduce_shape = list(x_v.shape)
        reduce_shape[self.axis] = 1
        return ex / np.sum(ex, axis=self.axis).reshape(reduce_shape)

def cross_entropy(predict : Node, label : Node, reduction : str = "mean"):
    __reductions__ = ["mean", "sum"]
    if reduction == "mean":
        return - reduce_mean(label * log(predict))
    elif reduction == "sum":
        return - reduce_sum(label * log(predict))
    else:
        raise Exception(f"reduction only receive {__reductions__}")

def mean_square_error(predict : Node, label : Node, reduction : str = "mean"):
    __reductions__ = ["mean", "sum"]
    if reduction == "mean":
        return reduce_mean((predict - label) ** 2)
    elif reduce_mean == "sum":
        return reduce_sum((predict - label) ** 2)
    else:
        raise Exception(f"reduction only receive {__reductions__}")


class Linear(object):
    def __init__(self, input_dim : int, output_dim : int, bias : bool = True, act : str = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if act and act not in _register_act_functions:
            raise ValueError(f"input activate function '{act}' is not in registered activate function list:{list(_register_act_functions.keys())}")
        self.act = act
        self.W = Variable(np.random.randn(input_dim, output_dim))
        if bias:
            self.b = Variable(np.random.randn(1, output_dim))
    
    def __call__(self, X : Node):
        if not isinstance(X, Node):
            raise ValueError("Linear's parameter X must be a Node!")
        out = X @ self.W + self.b
        if self.act:
            act_func = _register_act_functions[self.act]
            return act_func(out)
        else:
            return out

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

@_register_grad_functions("add")
def __add_gradient(op_node : Operation, grad : np.ndarray):
    return np.array([
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ])

@_register_grad_functions("minus")
def __minus_gradient(op_node : Operation, grad : np.ndarray):
    return np.array([
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        -1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ])

@_register_grad_functions("negative")
def __negative_gradient(op_node : Operation, grad : np.ndarray):
    return np.array([-1. * grad])

@_register_grad_functions("elementwise_pow")
def __elementwise_pow_gradeint(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.y
    return np.array([y * (x ** (y - 1)) * grad])

@_register_grad_functions("matmul")
def __matmul_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return np.array([grad @ y.T, x.T @ grad])

@_register_grad_functions("multiply")
def __multiply_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return np.array([y * grad, x * grad])

@_register_grad_functions("reduce_sum")
def __reduce_sum_gradient(op_node : Operation, grad : np.ndarray):
    return np.array([1. * grad])

@_register_grad_functions("reduce_mean")
def __reduce_mean_gradient(op_node : Operation, grad : np.ndarray):
    multiplier = op_node.input_nodes[0].data.size / op_node.data.size
    return np.array([1. / multiplier * grad])

@_register_grad_functions("log")
def __log_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    return np.array([1. / x * grad])

@_register_grad_functions("sigmoid")
def __sigmoid_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    ex = EXP(x)
    return np.array([ex / ((1 + ex) ** 2) * grad])

@_register_grad_functions("tanh")
def __tanh_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    e2x = EXP(2. * x)
    return np.array([4 * e2x / ((1 + e2x) ** 2) * grad])

@_register_grad_functions("relu")
def __relu_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x
    return np.array([k * grad])

@_register_grad_functions("leaky_relu")
def __leaky_relu(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.data
    k = y / x
    return np.array([k * grad])

@_register_grad_functions("elu")
def __elu_gradient(op_node : Operation, grad : np.ndarray):
    x = op_node.input_nodes[0].data
    k = np.array(x)
    k[k >= 0] = 1.
    k[k < 0] = op_node.alpha * np.exp(k[k < 0])
    return np.array([k * grad])

@_register_grad_functions("softmax")
def __softmax_gradient(op_node : Operation, grad : np.ndarray):
    f = op_node.data
    return f * (1 - f)

# create session to update nodes' data in the graph
class Session(object):
    def run(self, root_op : Operation, feed_dict : dict = {}):
        for node in _default_graph:
            if isinstance(node, Variable):
                node.data = np.array(node.data)
            elif isinstance(node, Placeholder):
                node.data = np.array(feed_dict[node])
            else:
                input_datas = [n.data for n in node.input_nodes]
                node.data = node.compute(*input_datas)
        return root_op

# optimizer
class Optimizer(object):
    def __init__(self, learning_rate : float = 1e-3):
        """
            base for all the optimizer
        """
        self.learning_rate = learning_rate
    
    def __backwards(self, op_node : Operation):
        """
            do the BP from the op_node, 
            return a gradient dict including op_node's gradients with respect to all the nodes before op_node
        """
        # wo will do the BP through BFS
        grad_table = {}
        grad_table[op_node] = 1.
        visit_nodes = set()
        queue = Queue()
        visit_nodes.add(op_node)
        queue.put(op_node)

        while not queue.empty():
            cur_node = queue.get()

            if cur_node != op_node:
                grad_table[cur_node] = 0.
                for next_node in cur_node.next_nodes:
                    grad_loss_wrt_next_node = grad_table[next_node]                                 # loss gradient of next_node
                    next_node_op_name = next_node.__class__.__name__                                # next_node must be an Operation, we get its name
                    gradient_func = _register_grad_functions[next_node_op_name]                     # get next_node's corresponding gradient function

                    grad_loss_wrt_cur_node = gradient_func(next_node, grad_loss_wrt_next_node)      # call the gradient function to get the sub-gradient
                    
                    if len(next_node.input_nodes) == 1:                                            # if next_node represents monocular operators, then add to total gradient directly
                        grad_table[cur_node] += grad_loss_wrt_cur_node
                    else:                                                                           # else get the portion size of gradient
                        cur_node_in_next_node_index = next_node.input_nodes.index(cur_node)
                        grad_table[cur_node] += grad_loss_wrt_cur_node[cur_node_in_next_node_index]

            if isinstance(cur_node, Operation):                                                     # put next op node into queue to do the BFS
                for input_node in cur_node.input_nodes:
                    if input_node not in visit_nodes:                                               # only add nodes which haven't been updated/visited yet
                        visit_nodes.add(input_node)
                        queue.put(input_node)

        return grad_table

    def minimize(self, loss_node : Operation):
        """
            concrete optimizer method, 
            this method will update parameters before "loss" node(include loss)
        """
        pass


class SGD(Optimizer):   # Stochastic gradient descent 
    def __init__(self, learning_rate : float = 1e-3):
        super().__init__(learning_rate=learning_rate)
    
    def minimize(self, loss_node : Operation):
        lr = self.learning_rate
        grad_table = self._Optimizer__backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                node.data -= lr * grad
                
        return grad_table


def view_graph(node : Operation, file_name : str = "./dot", format="png"):
    from graphviz import Digraph
    dot = Digraph(format=format)

    if not isinstance(node, Operation):
        raise ValueError("input node must be an Operation!")
    queue = Queue()
    visit = set()
    queue.put(node)
    visit.add(node)
    node2id = {}
    for index, n in enumerate(_default_graph):
        dot.node(name=str(index), label=str(n.__class__.__name__))
        node2id[n] = str(index)

    while not queue.empty():
        cur_node = queue.get()
        for input_node in cur_node.input_nodes:
            if isinstance(input_node, Operation) and input_node not in visit:   
                visit.add(input_node)
                queue.put(input_node)
            dot.edge(node2id[input_node], node2id[cur_node])

    dot.render(filename=file_name, cleanup=True)

if __name__ == "__main__":
    for k, v in _register_grad_functions.items():
        print("{:15}:{}".format(k, v))