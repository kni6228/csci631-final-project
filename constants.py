"""
Authors: Karthik Iyer (kni6228@rit.edu), Venkata Thanmai Mande (vm6710@rit.edu)
This file houses useful constants used throughout the code.
"""

from os.path import abspath

import torch

NUM_CLASSES = 88
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FEATURE_EXTRACT = True
DATASET_PATH = './dataset/{}'
LABELS_PATH = abspath('./dataset/labels_mapping.txt')
LEARNING_CURVE_PATH = './graphs/learning_curve.png'
MODEL_PATH = abspath('./model.pth')
