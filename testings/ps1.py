import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(), "salary.csv"), sep=",")
encoder = LabelEncoder() # Encoder


# First of all, who this is a tiny file, it doesn't matter what we are gonna do
Salary_Numpy_Y = df['Salary'].to_numpy()


X = np.arange(0,Salary_Numpy_Y.shape[0])

df['Position-cat'] = encoder.fit_transform(df['Position'])

print(df)








# Create the graph
fig, ax = plt.subplots(figsize=(5, 2.7),  layout='constrained')
ax.plot(X,Salary_Numpy_Y, )
ax.scatter(X,Salary_Numpy_Y, s=50, facecolor='red', edgecolor='k')
ax.legend()

plt.show()