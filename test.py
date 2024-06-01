import numpy as np
from pathlib import Path
import time

import torch
import torch.nn as nn

from utils.data import data_loaders
from utils.model import RandLANet

import laspy
import argparse
import os

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

def read_las(args):
    lasfile = laspy.read(args.data)
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
    
    return data, labels, xy_min

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LAS files and save as numpy arrays.")
    parser.add_argument("--data", required=True, help="Path to the LAS file.")
    parser.add_argument("--log_dir", required=True, help="Model folder.")
    parser.add_argument("--num_class", required=False, default=2, help="Number of classes.")
    return parser.parse_args()

def predict(data, labels, args):
    points = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print('Loading model...')

    d_in = 6
    num_classes = args.num_class
    
    model = RandLANet(d_in, num_classes, 16, 4, device)
    model.load_state_dict(torch.load('runs/'+args.log_dir+'/checkpoint_100.pth')['model_state_dict'])
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
        
    return predictions
    
if __name__ == "__main__":
    t0 = time.time()
    args = parse_arguments()
    print('Loading data...')
    data, labels, xy_min = read_las(args)
    predictions = predict(data, labels, args)

    t1 = time.time()
    # write point cloud with classes
    print('Assigning labels to the point cloud...')
    num_rows = data.shape[0]
    predictions = predictions.reshape(num_rows, 1)
    data[:, 0:2] += xy_min
    result = np.hstack((data, predictions))
    
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    filename = os.path.basename(str(args.data))
    filename = os.path.splitext(filename)[0]
    filename_pred = os.path.join(result_dir, filename + '_pred')
    save_las(result, filename_pred)

print('Done. Time elapsed: {:.1f}s'.format(t1-t0))
