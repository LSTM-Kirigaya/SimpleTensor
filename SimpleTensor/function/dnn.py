# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13
# TODO : add a class to program the operation which have paramter and can update its self

from SimpleTensor import Node, Variable, DnnOperator
from SimpleTensor import runtime, core
import numpy as np

class Linear(DnnOperator):
    def __init__(self, input_dim : int, output_dim : int, bias : bool = True, act : str = None, init: str=""):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if act and act not in runtime.activate_func:
            raise ValueError(f"input activate function '{act}' is not in registered activate function list:{list(runtime.activate_func.keys())}")
        
        # TODO : consider a clever solution to initialize parameters
        # Now, it is unstable
        # https://www.cnblogs.com/wxkang/p/15366535.html
        self.W = Variable(np.random.randn(input_dim, output_dim), node_name=self.cur_name)
        if bias:
            self.b = Variable(np.random.randn(1, output_dim), node_name=self.cur_name)
        
        self.act = act
    
    def __call__(self, X : Node) -> Node:
        if not isinstance(X, Node):
            raise ValueError("Linear's parameter X must be a Node!")
        
        out = core.matmul(X, self.W, node_name=self.cur_name)
        out = core.add(out, self.b, node_name=self.cur_name)
        
        if self.act:
            act_func = runtime.activate_func[self.act]
            return act_func(out, node_name=self.cur_name)
        else:
            return out

class Conv2D(DnnOperator):
    ...

class Rnn(DnnOperator):
    ...