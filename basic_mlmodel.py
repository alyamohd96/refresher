# import library
import pandas as pd


# Load CSV using pandas
#   Why pandas? recommended. flexible. can start summarizing and plotting immediately
filename = 'iris.csv'
iris_dt = pd.read_csv(filename)
print(iris_dt.shape)

# Descriptive Statistics