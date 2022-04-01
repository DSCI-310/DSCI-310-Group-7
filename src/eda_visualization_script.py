"""
Creating a exploratory histogram of the dataset.

Usage: src/eda_visualization_script.py data_path out_file

Options:  
data_path   The path of the dataset
out_file    The local file path to save the figure
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Exploratory data visualization')
parser.add_argument('data_path', type=str, help='the path of the dataset')
parser.add_argument('out_file', type=str, help='the path of the saved figure')
args = parser.parse_args()

data_path = args.data_path
path = args.out_file

data = pd.read_csv(data_path)
data.hist(figsize=(15,15))
plt.suptitle("fig.1 Histogram of all features")
plt.savefig(path)