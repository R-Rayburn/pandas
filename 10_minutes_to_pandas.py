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

# Creates a dataframe copy with including a new column 'E'
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
# Sets the first two rows to 1 in 'E' column
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)

# Drop rows with missing data
df1.dropna(how='any')

# Fill in missing data
df1.fillna(value=5)

# boolean mask where values are nan
pd.isna(df1)

## Stats
## Operations in general exclude missing data

# descriptive statistic
df.mean()

# same operation on other axis
df.mean(1)

# Shifts a series
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)

# Subtracts a series from a dataframe
df.sub(s, axis='index')

# Applying a function to a DataFrame
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# Creating a histogram
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
s.value_counts()

# Series have some string processing methods in str attribute
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())

# Concatenating
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
# break into pieces
pieces = [df[:3], df[3:7], df[7:]]
print(pd.concat(pieces))

# NOTE
# ----
# Adding a column to a DF is fast, but adding a row requires a copy.
# It can be expensive to add a row.
# Recommendation is to pass in a pre-built list of records into the DF const
#  instead of building DF iteratively.

# Joining DataFrames
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
