import SimpleTensor as st
from SimpleTensor.view import view_graph

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_X = pd.read_csv("./train_feature.csv", header=None).to_numpy().astype("float32")
train_Y = pd.read_csv("./train_target.csv", header=None).to_numpy()
train_X = (train_X - train_X.min(axis=0)) / np.ptp(train_X, axis=0)

label = st.numpy_one_hot(train_Y)

X = st.Placeholder()
Y = st.Placeholder()
out = st.dnn.Linear(2, 2, act="sigmoid")(X)

loss = st.measure.CrossEntropy()(predict=out, label=Y)
session = st.Session()
optimizer = st.optimizer.SGD(learning_rate=5e-3)

losses = []
acces  = []
for epoch in range(10):
    session.run(root_op=loss, feed_dict={X : train_X, Y : label})
    optimizer.minimize(loss)
    pre_lab = np.argmax(out.numpy, axis=1)
    acc = accuracy_score(train_Y, pre_lab)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss: {loss.numpy} accuracy: {acc}")
    losses.append(loss.numpy)
    acces.append(acc)

plt.plot(losses, label="loss")
plt.plot(acces,  label="acc")
plt.legend()
plt.show()

view_graph(format="pdf", direction="LR", show_grad=True)