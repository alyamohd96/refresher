# import library
import pandas as pd


# Load CSV using pandas
#   Why pandas? recommended. flexible. can start summarizing and plotting immediately
filename = 'iris.csv'
iris_dt = pd.read_csv(filename)

#### DESCRIPTIVE STATISTICS
# 1) Peek at raw data
print(iris_dt.head(20))

# 2) Dimensions of the data
print(iris_dt.shape)

# 3) Data Types for each attribute
print(iris_dt.dtypes)
# convert Species column to string
# TO DO: figure out whether need to encode
iris_dt["Species"] = iris_dt["Species"].astype(str)
print(iris_dt.dtypes)

# 4) Descriptive Statistics: describe() function list:
#   1) Count
#   2) Mean
#   3) Std Dev
#   4) Min Value
#   5) 25th Percentile
#   6) 50th Percentile
#   7) 75th Percentile
#   8) Max Value
print(iris_dt.describe())

# 5) Class Distribution (Classification Only)
#  need to check whether the observation data is balanced
print(iris_dt.groupby("Species").size())

# 6) Correlations between attributes.
# highly correlated attributes can influence the performance of some ml models
# Pearson's Correlation Coefficient
print(iris_dt.corr(method='pearson'))

# 7) Skew of Univariate Distribution
# high skewness influences the machine learning model. leads to excessively large variance in estimates
print(iris_dt.skew())

### DATA VISUALIZATION