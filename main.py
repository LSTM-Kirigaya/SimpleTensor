import SimpleTensor as st
from SimpleTensor.function.measure import CrossEntropy
from SimpleTensor.optimizer import SGD
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

# df = pd.DataFrame({
#     "x1" : train_X[:, 0],
#     "x2" : train_X[:, 1],
#     "y" : train_Y.reshape(-1)
# })

# seaborn.scatterplot(data=df, x="x1", y="x2", hue="y")
# plt.show()

label = st.numpy_one_hot(train_Y)

X = st.Placeholder()
Y = st.Placeholder()

out1 = st.dnn.Linear(2, 2, act="sigmoid")(X)
# out2 = Linear(8, 16, act="sigmoid")(out1)
# out3 = Linear(16, 8)(out2)
# out4 = Linear(8, 2, act="sigmoid")(out3)


loss = CrossEntropy(reduction="mean")(predict=out1, label=Y)
session = st.Session()
optimizer = SGD(learning_rate=1e-2)

losses = []
acces  = []
for epoch in range(10):
    session.run(root_op=loss, feed_dict={X : train_X, Y : label})
    optimizer.minimize(loss)
    pre_lab = np.argmax(out1.numpy, axis=1)
    acc = accuracy_score(train_Y, pre_lab)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss: {loss.numpy} accuracy: {acc}")
    losses.append(loss.numpy)
    acces.append(acc)

plt.style.use("gadfly")
plt.plot(losses, label="loss")
plt.plot(acces,  label="acc")
plt.legend()
plt.show()

view_graph(format="pdf", direction="LR", show_grad=True)