import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(), "salary.csv"))
print(df)
