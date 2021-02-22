from SimpleTensor.core import Placeholder, Variable
from SimpleTensor.core import Session
from SimpleTensor.core import mean_square_error, SGD

from sklearn.datasets import load_boston 
import matplotlib.pyplot as plt
import numpy as np

X, Y = load_boston(return_X_y=True)

sample_num = X.shape[0]
ratio = 0.8
offline = int(ratio * sample_num)
indexes = np.arange(sample_num)
np.random.shuffle(indexes)

train_X, train_Y = X[indexes[:offline]], Y[indexes[:offline]].reshape([-1, 1])
test_X, test_Y = X[indexes[offline:]], Y[indexes[offline:]].reshape([-1, 1])

X = Placeholder()
Y = Placeholder()
W1 = Variable(np.random.randn(13, 4))
b1 = Variable(np.random.randn(1, 4))
W2 = Variable(np.random.randn(4, 1))
b2 = Variable(np.random.randn(1, 1))

out1 = X @ W1 + b1
out2 = out1 @ W2 + b2

loss = mean_square_error(predict=out2, label=Y)

session = Session()
optimizer = SGD(learning_rate=1e-7)

losses = []

for epoch in range(20):
    session.run(root_op=loss, feed_dict={X : train_X, Y : train_Y})
    losses.append(loss.numpy)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy}")

session.run(root_op=loss, feed_dict={X : test_X, Y : test_Y})
predict = out2.numpy

plt.style.use("seaborn")
plt.subplot(1, 2, 1)
plt.plot(losses, "-o")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(test_Y.reshape(-1), "r", label="ground truth", alpha=0.5)
plt.plot(predict.reshape(-1), "b", label="predict", alpha=0.5)
plt.legend()
plt.grid(True)


plt.show()