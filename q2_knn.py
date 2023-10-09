import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

def knn(train_x, train_y, test_x, test_y, k=1):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for t in range(len(test_x)):
        d_min = np.array([np.inf for i in range(k)])
        labels = np.array([-1 for i in range(k)])
        for i in range(len(train_x)):
            max_id = np.argmax(d_min)
            d = ((train_x[i]-test_x[t])**2).sum()
            if d < d_min[max_id]:
                d_min[max_id] = d
                labels[max_id] = train_y[i]
        pred = np.bincount(labels).argmax()
        
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

    print("for k =", k)
    print("accuracy:", acc)
    print("precision:", pre)
    print("recall:", rec, '\n')

    return acc, pre, rec


if __name__ == "__main__":
    data = pd.read_csv("./hw3Data/emails.csv")
    # print(data.iloc[:, 1:-1].head())
    X = data.iloc[:, 1:-1].to_numpy()
    Y = data.iloc[:, -1].to_numpy()
    # print(X)

    iters = 5

    x_test = np.array([X[:1000], X[1000:2000], X[2000:3000], X[3000:4000], X[4000:5000]])
    x_train = np.array([np.delete(X, slice(0, 1000), axis=0), np.delete(X, slice(1000, 2000), axis=0), np.delete(X, slice(2000, 3000), axis=0), np.delete(X, slice(3000, 4000), axis=0), \
                        np.delete(X, slice(4000, 5000), axis=0)])
    y_test = np.array([Y[:1000], Y[1000:2000], Y[2000:3000], Y[3000:4000], Y[4000:5000]])
    y_train = np.array([np.delete(Y, slice(0, 1000), axis=0), np.delete(Y, slice(1000, 2000), axis=0), np.delete(Y, slice(2000, 3000), axis=0), np.delete(Y, slice(3000, 4000), axis=0), \
                        np.delete(Y, slice(4000, 5000), axis=0)])
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    K = [1, 3, 5, 7, 10]
    avg_acc = []
    for k in K:
        for f in range(iters):
            temp = []
            a, p, r = knn(x_train[f], y_train[f], x_test[f], y_test[f], k=k)
            temp.append(a)
        avg_acc.append(mean(temp))
    print(avg_acc)
    plt.plot(K, avg_acc)
    plt.xlabel("k")
    plt.ylabel("Average accuracy")
    plt.title("KNN 5-fold cross validation")
    plt.savefig("knn.pdf")
    


        

