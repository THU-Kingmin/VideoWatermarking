import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

output_vis_list = [torch.randn(1, 356*1024) for _ in range(100)]
data = torch.cat([x.view(1, -1) for x in output_vis_list], dim=0)
garble = torch.randint(0, 2, (100,1))

print(data.shape)
print(garble.shape)

data_np = data.numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_np)
plt.figure()
for i in range(data_np.shape[0]):
    color = 'r' if garble[i] == 0 else 'b'
    plt.scatter(data_np[i, 0], data_np[i, 1], marker='o', color=color)

plt.savefig('output.png')