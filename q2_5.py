import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./hw3Data/emails.csv")
# print(data.iloc[:, 1:-1].head())
X = data.iloc[:, 1:-1].to_numpy()
Y = data.iloc[:, -1].to_numpy()
# print(X)

x_test = X[:1000]
x_train = np.delete(X, slice(0, 1000), axis=0)
y_test = Y[:1000]
y_train = np.delete(Y, slice(0, 1000), axis=0)

print("KNN starting...")
k = 5
knn_conf = []
knn_pred = []
for t in range(len(x_test)):
    d_min = np.array([np.inf for i in range(k)])
    labels = np.array([-1 for i in range(k)])
    for i in range(len(x_train)):
        max_id = np.argmax(d_min)
        d = ((x_train[i]-x_test[t])**2).sum()
        if d < d_min[max_id]:
            d_min[max_id] = d
            labels[max_id] = y_train[i]
    counts = np.bincount(labels)
    pred = counts.argmax()
    knn_pred.append(pred)
    conf = (counts.sum() - counts[0]) / (counts.sum())
    knn_conf.append(conf)

print("logistic regression starting...")
log_conf = []
log_pred = []
n, p = x_train.shape[0], x_train.shape[1]
train_bias = np.ones((n, 1))
test_bias = np.ones((x_test.shape[0], 1))
x_train = np.hstack((train_bias, x_train))
x_test = np.hstack((test_bias, x_test))
w = np.zeros(p+1)

for steps in range(2000):    
    h = 1 / (1 + np.exp(-np.matmul(x_train, w)))
    L = (1/n) * (-x_train * (y_train - h).reshape(n, 1)).sum(axis=0)
    w -= 0.001 * L

for t in range(len(x_test)):
    log_pred.append(int(1 / (1 + np.exp(-np.matmul(x_test[t], w))) >= 0.5))
    log_conf.append(1 / (1 + np.exp(-np.matmul(x_test[t], w))))

print("plotting ROC...")
threshold = np.arange(0, 1, 0.001)
tpr_knn, tpr_log = [], []
fpr_knn, fpr_log = [], []
label = y_test

for x in threshold:
    label_new_knn = [True if i>=x else False for i in knn_conf]
    label_new_log = [True if i>=x else False for i in log_conf]
    tp_knn, tn_knn, fp_knn, fn_knn = 0, 0, 0, 0
    tp_log, tn_log, fp_log, fn_log = 0, 0, 0, 0
    for i in range(len(y_test)):
        if label[i] == 1:
            if label_new_knn[i] is True:
                tp_knn+=1
            else:
                fn_knn+=1

            if label_new_log[i] is True:
                tp_log+=1
            else:
                fn_log+=1
               
        else:
            if label_new_knn[i] is True:
                fp_knn+=1
            else:
                tn_knn+=1

            if label_new_log[i] is True:
                fp_log+=1
            else:
                tn_log+=1
        
    tpr_knn.append(tp_knn / (tp_knn + fn_knn))
    fpr_knn.append(fp_knn / (tn_knn + fp_knn))

    tpr_log.append(tp_log / (tp_log + fn_log))
    fpr_log.append(fp_log / (tn_log + fp_log))

plt.plot(fpr_knn, tpr_knn, label='KNN')
plt.plot(fpr_log, tpr_log, label='Logistic Regression')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("./Q5_ROC.pdf")
plt.show()