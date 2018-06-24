from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

raw = pd.read_csv("./file.txt").as_matrix()
pca = pd.read_csv("./output.txt").as_matrix()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(raw[:,0], raw[:,1], raw[:,2])
fig.savefig("raw.png")

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(pca[:,0], pca[:,1])
fig.savefig("downprojected.png")
