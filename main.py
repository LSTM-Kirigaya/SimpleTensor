from SimpleTensor.core import Placeholder, Variable
from SimpleTensor.core import Session
from SimpleTensor.core import mean_square_error, SGD, Linear
from SimpleTensor.core import view_graph

from sklearn.datasets import load_boston 
import matplotlib.pyplot as plt
import numpy as np

X, Y = load_boston(return_X_y=True)

np.random.seed(123)
sample_num = X.shape[0]
ratio = 0.8
offline = int(ratio * sample_num)
indexes = np.arange(sample_num)
np.random.shuffle(indexes)

train_X, train_Y = X[indexes[:offline]], Y[indexes[:offline]].reshape([-1, 1])
test_X, test_Y = X[indexes[offline:]], Y[indexes[offline:]].reshape([-1, 1])

X = Placeholder()
Y = Placeholder()

out1 = Linear(13, 8, act="relu")(X)
out2 = Linear(8, 4, act=None)(out1)
out3 = Linear(4, 1, act=None)(out2)

loss = mean_square_error(predict=out3, label=Y)

session = Session()
optimizer = SGD(learning_rate=1e-8)

losses = []

for epoch in range(30):
    session.run(root_op=loss, feed_dict={X : train_X, Y : train_Y})
    losses.append(loss.numpy)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy}")

session.run(root_op=loss, feed_dict={X : test_X, Y : test_Y})
predict = out3.numpy

plt.style.use("seaborn")
plt.subplot(1, 2, 1)
plt.plot(losses, "-o")
plt.xlabel("number of iteration")
plt.ylabel("mean loss")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(test_Y.reshape(-1), "r", label="ground truth", alpha=0.5)
plt.plot(predict.reshape(-1), "b", label="predict", alpha=0.5)
plt.xlabel("sample id")
plt.ylabel("price")
plt.legend()
plt.grid(True)

plt.show()

# from SimpleTensor.core import _default_graph
# for item in _default_graph:
#     print(item.__class__)

# view_graph(node=loss, format="pdf")