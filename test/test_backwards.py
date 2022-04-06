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

st.view.view_graph(format="pdf", direction="LR", show_grad=True)