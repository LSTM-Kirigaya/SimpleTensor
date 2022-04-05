import SimpleTensor as st
from SimpleTensor.view import view_graph

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

train_X = pd.read_csv("./train_feature.csv", header=None).to_numpy().astype("float32")
train_Y = pd.read_csv("./train_target.csv", header=None).to_numpy()
train_X = (train_X - train_X.min(axis=0)) / np.ptp(train_X, axis=0) 

label = st.numpy_one_hot(train_Y)

X = st.Data(data=train_X)
Y = st.Data(data=train_Y)

out1 = st.dnn.Linear(2, 8)(X)
out2 = st.dnn.Linear(8, 2, act="sigmoid")(out1)

loss = st.measure.CrossEntropy(reduction="mean")(predict=out2, label=Y)
session = st.Session()
optimizer = st.optimizer.SGD(learning_rate=1e-3)

# TODO : try to wrap session and optimizer together
losses = []
acces  = []
for epoch in range(20):
    session.run(root_op=loss)
    optimizer.minimize(loss)

    pre_lab = np.argmax(out2.numpy, axis=1)
    print(out2.numpy[0])
    acc = accuracy_score(train_Y, pre_lab)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss: {loss.numpy} accuracy: {acc}")
    losses.append(loss.numpy)
    acces.append(acc)

# plt.style.use("gadfly")
# plt.plot(losses, label="loss")
# plt.plot(acces,  label="acc")
# plt.legend()
# plt.show()

# view_graph(format="pdf", direction="LR", show_grad=True)