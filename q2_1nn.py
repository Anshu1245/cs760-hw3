import numpy as np
import matplotlib.pyplot as plt


with open('./hw3Data/D2z.txt', 'r') as f:
    data = []
    all_text = f.readlines()
    for line in all_text:
        line = line.split(" ")
        line[-1] = line[-1][:-1]
        line = [float(i) for i in line]
        data.append(line)


data = np.array(data)

fig, ax = plt.subplots()

x = np.arange(-2, 2.1, 0.1)
test = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
pred = []
for p in test:
    dist_min = np.inf
    idx = -1
    for i in range(len(data)):
        new_dist = (p[0]-data[i, 0])**2 + (p[1]-data[i, 1])**2
        if new_dist<dist_min:
            dist_min = new_dist
            idx = int(data[i, -1])
    pred.append(idx)
    if idx == 0:
        plt.plot(p[0], p[1], 'ro', markersize=2)
    else:
        plt.plot(p[0], p[1], 'bo', markersize=2)

ax.scatter(data[:, 0][data[:, -1]==0],data[:, 1][data[:, -1]==0], c="r", marker='x', s=70)
ax.scatter(data[:, 0][data[:, -1]==1],data[:, 1][data[:, -1]==1], c="b", marker='o', s=70)
plt.xlabel("X0")
plt.ylabel("X1")



plt.show()

















