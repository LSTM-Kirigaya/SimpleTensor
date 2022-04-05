# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.22
# caption : used to test if the auto derivation system works correctly

import sys, os
sys.path.append(os.path.abspath('.'))

import SimpleTensor as st

x = st.Variable(1.0)
y = st.Variable(4.0)
z = st.Variable(3.0)

w = z * (x + y)

session = st.Session()
session.run(w)
optim = st.optimizer.SGD()
grad_table = optim.backwards(w)
print(grad_table[x])
print(grad_table[y])
print(grad_table[z])
print(grad_table[w])