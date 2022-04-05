# -*- encoding:utf-8 -*-
# author: Zhelong Huang
# version: 1.0.2
# date : 2022.3.21
# caption : used to measure the stability of training

import sys, os
sys.path.append(os.path.abspath('.'))

import SimpleTensor as st
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', type=int, default=10,
    help="times of testing"
)
parser.add_argument(
    '--epoch', type=int, default=10,
    help="epoch of training"
)
parser.add_argument(
    '--init', type=str, default="",
    help="init method of Linear"
)

args = vars(parser.parse_args())
N     = args["n"]
EPOCH = args['epoch']
INIT  = args["init"]

train_X = pd.read_csv("./train_feature.csv", header=None).to_numpy().astype("float32")
train_Y = pd.read_csv("./train_target.csv", header=None).to_numpy()
train_X = (train_X - train_X.min(axis=0)) / np.ptp(train_X, axis=0) 
label = st.numpy_one_hot(train_Y)
X = st.Placeholder()
Y = st.Placeholder()        
optimizer = st.optimizer.SGD(learning_rate=1e-2)
linear = st.dnn.Linear(2, 2, act="sigmoid", init=INIT)
out = linear(X)
loss = st.measure.CrossEntropy(reduction="mean")(predict=out, label=Y)
session = st.Session()

def main():
    total_loss_ratio = []
    total_acc_ratio = []
    for _ in range(N):
        losses = []
        acces  = []
        loss_ratio = []
        acc_ratio = []
        for epoch in range(EPOCH):
            session.run(root_op=loss, feed_dict={X : train_X, Y : label})
            optimizer.minimize(loss)
            pre_lab = np.argmax(out.numpy, axis=1)
            acc = accuracy_score(train_Y, pre_lab)

            if len(losses) > 0:
                loss_ratio.append(loss.numpy - losses[-1])
                acc_ratio.append(acc - acces[-1])
            losses.append(loss.numpy)
            acces.append(acc)
        total_loss_ratio.append(np.mean(loss_ratio))
        total_acc_ratio.append(np.mean(acc_ratio))
        linear.reset_params()

    print("acc increase ratio: ", np.mean(total_acc_ratio))
    print("loss increase ratio: ", np.mean(total_loss_ratio))

main()