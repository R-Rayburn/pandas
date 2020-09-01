import numpy as np
import pandas as pd

# Creating a series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Creating a Dataframe of dates from an array
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

# Creating a Dataframe from an object
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

# List column dtypes
print(df2.dtypes)

# Viewing the top of the frame
print(df.head())

# Viewing the bottom 3 of the frame
print(df.tail(3))

# Display the index
print(df.index)

# Display the columns
print(df.columns)

# Convert to NumPy representation
# NOTE: can be expensive when having DF with different data types
# floating-point values: fast
print(df.to_numpy())
# multiple dtypes: expensive
print(df2.to_numpy())

# Statistic summary containing:
#  count, mean, std, min, 25%, 50%, 75%, max
print(df.describe())

# Transpose
print(df.T)

# Sorting by axis
print(df.sort_index(axis=1, ascending=False))

# Sorting by values
print(df.sort_values(by='B'))

# Selecting a single column, yeilding a Series
print(df['A'])
print(df.A)

# Slicing by row index
print(df[0:3])

# Slicing by row value
print(df['20130102':'20130104'])

# Selecting by label
print(df.loc[dates[0]])

# Selecting multi axis by label
print(df.loc[:, ['A', 'B']])

# Label slicing
print(df.loc['20130102':'20130104', ['A', 'B']])

# Label slicing that returns a series by reducing a dimension
print(df.loc['20130102', ['A', 'B']])

# Getting a scalar value
print(df.loc[dates[0], 'A'])

# Fast access scalar value
print(df.at[dates[0], 'A'])

# Select by position
print(df.iloc[3])

# Position slicing
print(df.iloc[3:5, 0:2])

# Select by list of integers
print(df.iloc[[1, 2, 4], [0, 2]])

# Slicing rows
print(df.iloc[1:3, :])

# Slicing columns
print(df.iloc[:, 1:3])

# Getting scalar value
print(df.iloc[1, 1])

# Fast access scalar
print(df.iat[1, 1])

# Using a column's value to select data
print(df[df['A'] > 0])

# Selecting values in a DF where condition is met
# Values that don't meet condition come back as NaN
print(df[df > 0])

# Filtering with isin()
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2[df2['E'].isin(['two', 'four'])])

# Setting a new column
s1 = pd.Series([1, 2, 3, 4, 5, 6],
               index=pd.date_range('20130102', periods=6))
df['F'] = s1

# Setting values by label
df.at[dates[0], 'A'] = 0

# Setting values by position
df.iat[0, 1] = 0

# Assigning column with an array
df.loc[:, 'D'] = np.array([5] * len(df))

# Setting to negative value using WHERE operation
df2 = df.copy()
df2[df2 > 0] = -df2


