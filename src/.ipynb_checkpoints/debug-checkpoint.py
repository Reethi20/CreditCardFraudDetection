import pandas as pd

print('--- Starting Debug Script (reading 100 rows) ---')

try:
    df = pd.read_csv('data/creditcard.csv', nrows=100)
    print('Successfully loaded the first 100 rows of data/creditcard.csv')
    print(f'Dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
except Exception as e:
    print(f'An error occurred while loading the data: {e}')

print('--- Debug Script Finished ---')
