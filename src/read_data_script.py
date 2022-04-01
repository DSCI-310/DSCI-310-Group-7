"""
Download the dataset from the url online and save it to the local path.

Usage: src/read_data_script.py url out_file

Options:  
url         The url of the original dataset
out_file    The local file path to save the processed data including the file type
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Read and save dataset')
parser.add_argument('url', type=str, help='the url of the dataset')
parser.add_argument('out_file', type=str, help='the path of the saved dataset locally include the file type')
args = parser.parse_args()

url = args.url
path = args.out_file

data = pd.read_csv(url, header=None)
data.to_csv(path)

