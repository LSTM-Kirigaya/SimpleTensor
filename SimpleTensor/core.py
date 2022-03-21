# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

import numpy as np
from typing import Union, List
import abc
from SimpleTensor.constant import runtime

from SimpleTensor.util import back_print, fore_print
    
class Node            : ...
class add             : ...
class minus           : ...
class negative        : ...
class multiply        : ...
class matmul          : ...
class elementwise_pow : ...


class Node(abc.ABC):
    # __slots__ = "next_nodes", "data", "node_name"
    def __init__(self, node_name: str=""):
        """
            base for all types of node in the static calculation graph
        """
        self.next_nodes = []
        self.data = None
        self.node_name = node_name
        runtime.global_calc_graph.append(self)
    
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
        return "{}({})".format(self.__class__.__name__, str(self.data))
    
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

class Operation(Node, abc.ABC):
    def __init__(self, input_nodes : List[Node] = [], node_name: str=""):
        super().__init__(node_name=node_name)
        self.input_nodes = input_nodes
        for node in input_nodes:
            node.next_nodes.append(self)
    
    @abc.abstractmethod
    def compute(self, *args): ...

class Placeholder(Node): ...

class Variable(Node):
    def __init__(self, init_value : Union[np.ndarray, list] = None, node_name:str=""):
        super().__init__(node_name=node_name)
        self.data = init_value


# create session to update nodes' data in the graph
class Session(object):
    def run(self, root_op : Operation, feed_dict : dict = {}):
        for node in runtime.global_calc_graph:
            if isinstance(node, Variable):
                node.data = np.array(node.data)
            elif isinstance(node, Placeholder):
                try:
                    node.data = np.array(feed_dict[node])
                except:
                    print(feed_dict.keys())
                    exit(-1)
            else:
                input_datas = [n.data for n in node.input_nodes]
                node.data = node.compute(*input_datas)
        return root_op

class DnnOperator(abc.ABC):
    def __init__(self) -> None:
        self.cur_name = self.__class__.__name__ + "_" + str(runtime.dnn_cnt[self.__class__.__name__])
        runtime.dnn_cnt[self.__class__.__name__] += 1

    @abc.abstractmethod
    def __call__(self) -> Node: ...

class DnnVarOperator(DnnOperator):
    @abc.abstractmethod
    def reset_params(self) -> None:  ...

    @abc.abstractmethod
    def get_params(self, *args, **kwargs) -> np.ndarray: ...

# ==============================
# basic function used in core.py
# ==============================
class add(Operation):
    def __init__(self, x : Node, y : Node, node_name: str=""):
        super().__init__(input_nodes=[x, y], node_name=node_name)
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v + y_v

class minus(Operation):
    def __init__(self, x : Node, y : Node, node_name: str=""):
        super().__init__(input_nodes=[x, y], node_name=node_name)
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v - y_v

class negative(Operation):
    def __init__(self, x : Node, node_name: str=""):
        super().__init__(input_nodes=[x], node_name=node_name)
    
    def compute(self, x_v : np.ndarray):
        return -1. * x_v

class elementwise_pow(Operation):
    def __init__(self, x : Node, y : Union[int, float], node_name: str=""):
        super().__init__(input_nodes=[x], node_name=node_name)
        self.y = y
    
    def compute(self, x_v : np.ndarray):
        return x_v ** self.y

class matmul(Operation):
    def __init__(self, x : Node, y : Node, node_name: str=""):
        super().__init__(input_nodes=[x, y], node_name=node_name)
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v @ y_v

class multiply(Operation):
    def __init__(self, x : Node, y : Node, node_name: str=""):
        super().__init__(input_nodes=[x, y], node_name=node_name)
    
    def compute(self, x_v : np.ndarray, y_v : np.ndarray):
        return x_v * y_v

class reduce_sum(Operation):
    def __init__(self, x : Node, axis : int = None, node_name: str=""):
        super().__init__(input_nodes=[x], node_name=node_name)
        self.axis = axis
    
    def compute(self, x_v : np.ndarray):
        return np.sum(x_v, axis=self.axis)

class reduce_mean(Operation):
    def __init__(self, x : Node, axis : int = None, node_name: str=""):
        super().__init__(input_nodes=[x], node_name=node_name)
        self.axis = axis
    
    def compute(self, x_v : np.ndarray):
        return np.mean(x_v, axis=self.axis)

class log(Operation):
    def __init__(self, x : Node, node_name: str=""):
        super().__init__(input_nodes=[x], node_name=node_name)
    
    def compute(self, x_v : np.ndarray):
        if (x_v <= 0).any():
            back_print("Oops, invalid value encountered in 'log', I guess you forget activation function", color="yellow")
        return np.log(x_v)