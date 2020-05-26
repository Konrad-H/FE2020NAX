# DATA EXTRACION FROM gefcom

import pandas as pd 
import os
import sys

# data = pd.read_csv("example.csv") 
# data.head()

current_dir = os.path.dirname(__file__)

oneup_dir = os.path.join(current_dir, os.path.pardir)

file_path = os.path.join(oneup_dir, "./gefcom.csv")
df = pd.read_csv(file_path) 
print(df.head())

# with  as f: