import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
import numpy as np
import argparse

def mean_std_2(mean, std):
    n=3
    mean_v = (mean[0]+mean[1])/2
    std_v = np.sqrt(((n - 1) * std[0]**2 + (n - 1) * std[1]**2) / (n + n - 2))
    print(mean_v, std_v)
    return mean_v, std_v

def mean_std_3(mean, std):
    n=3
    mean_v = (mean[0]+mean[1]+mean[2])/3
    variances = [s**2 for s in std]
    pooled_variance = np.mean(variances)
    overall_std = np.sqrt(pooled_variance)
    print(mean_v, overall_std)
    return mean_v, overall_std

data =  [76.80, 2.51, 48.03, 2.96, 88.90, 1.29]
mean = [data[0], data[2], data[4]]
std = [data[1], data[3], data[5]]

mean_std_3(mean, std)