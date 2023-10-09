import numpy as np
import matplotlib.pyplot as plt


# question number 5(a)

conf = [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7, 0.8, 0.85, 0.95]
label = [bool(i) for i in [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]]

thresh = np.arange(0, 1, 0.01)

tpr = []
fpr = []

for x in thresh:
    label_new = [True if i>=x else False for i in conf]
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(10):
        if label[i] is True:
            if label_new[i] is True:
                tp+=1
            else:
                fn+=1
        else:
            if label_new[i] is True:
                fp+=1
            else:
                tn+=1
    
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (tn + fp))
print(set(tpr), set(fpr))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("./ROC.pdf")
plt.show()

