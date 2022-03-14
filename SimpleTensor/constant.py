# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.13

import numpy as np
from SimpleTensor.util import Register
from collections import defaultdict

# constant variable
class Clip:
    PRECISE_LOW = 1e-127
    PRECISE_HIGH = 1e128
    EXP_PRECISE_LOW = -292.42
    EXP_RPECISE_HIGH = 294.73
    EXP = lambda x : np.exp(np.clip(x, Clip.EXP_PRECISE_LOW, Clip.EXP_RPECISE_HIGH))

# runtime variable
class runtime:
    activate_func = Register()
    gradient_func = Register()
    global_calc_graph = list()
    dnn_cnt = defaultdict(int)
    grad_table = None