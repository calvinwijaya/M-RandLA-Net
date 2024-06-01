import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

from utils.data import data_loaders
from utils.model import RandLANet

# Function to read and save las data
import laspy

def save_las(X, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    las.classification = X[:, 6]
    las.write(filename + ".las")

t0 = time.time()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Loading data...')
las_file = 'data/02_Test Kecil Full Data.las'

lasfile = laspy.read(las_file)
x = lasfile.x
y = lasfile.y
z = lasfile.z
r = lasfile.red
g = lasfile.green
b = lasfile.blue
C = lasfile.classification
data = np.column_stack((x, y, z, r, g, b))
labels = np.column_stack((C))

xy_min = np.amin(data, axis=0)[0:2]
data[:, 0:2] -= xy_min

points = torch.from_numpy(data)
labels = torch.from_numpy(labels)

print('Loading model...')

d_in = 6
num_classes = 2

model = RandLANet(d_in, num_classes, 16, 4, device)
model.load_state_dict(torch.load('runs/test/checkpoint_10.pth')['model_state_dict'])
model.eval()


print('Predicting labels...')
with torch.no_grad():
    points = points.to(device)
    points = points.float()
    points = points.unsqueeze(-1)
    points = points.permute(2, 0, 1)
    labels = labels.to(device)
    scores = model(points)
    predictions = torch.max(scores, dim=-2).indices
    accuracy = (predictions == labels).float().mean() # TODO: compute mIoU usw.
    print('Accuracy:', accuracy.item())
    predictions = predictions.cpu().numpy()

print('Writing results...')

t1 = time.time()
# write point cloud with classes
print('Assigning labels to the point cloud...')
num_rows = data.shape[0]
predictions = predictions.reshape(num_rows, 1)
data[:, 0:2] += xy_min
result = np.hstack((data, predictions))
save_las(result, 'hasil')

print('Done. Time elapsed: {:.1f}s'.format(t1-t0))
