# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.21
# caption : used to test BP
import sys, os
sys.path.append(os.path.abspath('.'))
import SimpleTensor as st

X = st.Variable(1.0)
Y = st.Variable(4.0)
Z = st.Variable(3.0)
W = Z * (X + Y)

session = st.Session()
session.run(W)

grad_table = st.optimizer.backwards(W)
print("X grad: ", grad_table[X])
print("Y grad: ", grad_table[Y])
print("Z grad: ", grad_table[Z])
print("W grad: ", grad_table[W])

# X grad:  3.0
# Y grad:  3.0
# Z grad:  5.0
# W grad:  1.0

# st.view.view_graph(format="pdf", direction="LR", show_grad=True)

st.runtime.global_calc_graph = []
X = st.Variable([[1.2, 1.5], [3.0, 6.0]])
Y = st.Variable([[1.0, 2.0]])

Z = st.reduce_sum(Y @ X)

session = st.Session()
session.run(Z)

grad_table = st.optimizer.backwards(Z)
print("X grad: ", grad_table[X])
print("Y grad: ", grad_table[Y])
print("Z grad: ", grad_table[Z])