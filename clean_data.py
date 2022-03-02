import pandas as pd

df = pd.read_csv("census.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.columns)

# remove whitspace of columnames
list_cols = df.columns.tolist()
list_cols = [col.strip() for col in list_cols]
df.columns = list_cols
print(df.columns)

# data contains columns with ' ?' value
# remove these rows
list_cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                 'race', 'sex', 'native-country']
print("Data shape before dropping rows: ", df.shape)
for col in list_cat_cols:
    df = df.drop(df[df[col].str.contains('?', regex=False)].index)

print("Data shape after dropping rows: ", df.shape)

# save data as processed csv
df.to_csv('processed_census.csv')
