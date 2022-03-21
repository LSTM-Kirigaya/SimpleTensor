# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

from SimpleTensor import Node, DnnOperator
from SimpleTensor import log, reduce_mean, reduce_sum
from SimpleTensor import core
import numpy as np

class CrossEntropy(DnnOperator):
    __reduction__ = {
        "mean" : reduce_mean,
        "sum" : reduce_sum
    }
    def __init__(self, reduction: str="sum") -> None:
        super().__init__()
        if reduction not in self.__reduction__:
            raise ValueError("{} not in avaliable reduction function:{}".format(reduction, self.__reduction__))
        self.reduction = reduction
        
    def __call__(self, predict: Node, label: Node):
        p_pre = log(predict, node_name=self.cur_name)
        p_pre = core.multiply(label, p_pre, node_name=self.cur_name)
        reduce_p = self.__reduction__[self.reduction](p_pre, node_name=self.cur_name)
        return core.negative(reduce_p, node_name=self.cur_name)

# TODO : cross_entropy seems to go wrong
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