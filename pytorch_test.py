// create pytorch cnn code

def add_two_numbers(a, b):
    return a + b

def find_odd_one_value():
    a = [1, 2, 3, 4, 5, 6, 7]
    for i in a:
        if i % 2 == 1:
            return i
    
    
## Create a pytorch model for MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

import os
import sys
import argparse
import logging

from torch.utils.tensorboard import SummaryWriter





