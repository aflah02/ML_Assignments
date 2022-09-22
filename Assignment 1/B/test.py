import pandas as pd
df = pd.read_csv('Real estate.csv')
# Since the dataset was released 4 years ago I assume the dates are relative to that and hence break the transaction date into years since last sold
df['Year last sold'] = df['X1 transaction date'].apply(lambda x: int(str(x).split('.')[0]))
df['Years since last sold'] = 2018 - df['Year last sold']
X = df.drop(columns=['Y house price of unit area', 'X1 transaction date', 'No', 'Year last sold'])
print(X)