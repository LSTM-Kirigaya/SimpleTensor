# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

__version__ = "1.0.1"

from SimpleTensor.constant import runtime, Clip
from SimpleTensor.util import numpy_one_hot
from SimpleTensor.core import Node, Operation, Variable, Placeholder, DnnOperator, DnnVarOperator, Data
from SimpleTensor.core import Session
from SimpleTensor.core import reduce_mean, reduce_sum, log
from SimpleTensor import optimizer
from SimpleTensor.function import dnn

# finish init and register, they are available in <runtime>
import SimpleTensor.function.activate
import SimpleTensor.function.gradient
import SimpleTensor.initialize

from SimpleTensor import view
from SimpleTensor.function import measure, activate