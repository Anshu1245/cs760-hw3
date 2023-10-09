import numpy as np
import pandas as pd


def logistic_regression(train_x, train_y, test_x, test_y, lr=0.001):
    n, p = train_x.shape[0], train_x.shape[1]
    train_bias = np.ones((n, 1))
    test_bias = np.ones((test_x.shape[0], 1))
    train_x = np.hstack((train_bias, train_x))
    test_x = np.hstack((test_bias, test_x))
    w = np.zeros(p+1)
    
    for steps in range(2000):    
        h = 1 / (1 + np.exp(-np.matmul(train_x, w)))
        L = (1/n) * (-train_x * (train_y - h).reshape(n, 1)).sum(axis=0)
        w -= lr * L

    tp, fp, tn, fn = 0, 0, 0, 0
    for t in range(len(test_x)):
        pred = int(1 / (1 + np.exp(-np.matmul(test_x[t], w))) >= 0.5)
        
        if test_y[t] == 0:
            if pred == 0:
                tn += 1
            else:
                fp += 1
        else:
            if pred == 0:
                fn += 1
            else:
                tp += 1

    acc = (tp + tn) / (tp + fn + fp + tn)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    
    print("accuracy:", acc)
    print("precision:", pre)
    print("recall:", rec, '\n')

    return acc, pre, rec


if __name__ == '__main__':
    data = pd.read_csv("./hw3Data/emails.csv")
    # print(data.iloc[:, 1:-1].head())
    X = data.iloc[:, 1:-1].to_numpy()
    Y = data.iloc[:, -1].to_numpy()
    # print(X)

    folds = 5

    x_test = np.array([X[:1000], X[1000:2000], X[2000:3000], X[3000:4000], X[4000:5000]])
    x_train = np.array([np.delete(X, slice(0, 1000), axis=0), np.delete(X, slice(1000, 2000), axis=0), np.delete(X, slice(2000, 3000), axis=0), np.delete(X, slice(3000, 4000), axis=0), \
                        np.delete(X, slice(4000, 5000), axis=0)])
    y_test = np.array([Y[:1000], Y[1000:2000], Y[2000:3000], Y[3000:4000], Y[4000:5000]])
    y_train = np.array([np.delete(Y, slice(0, 1000), axis=0), np.delete(Y, slice(1000, 2000), axis=0), np.delete(Y, slice(2000, 3000), axis=0), np.delete(Y, slice(3000, 4000), axis=0), \
                        np.delete(Y, slice(4000, 5000), axis=0)])
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    for f in range(folds):
        a, p, r = logistic_regression(x_train[f], y_train[f], x_test[f], y_test[f])