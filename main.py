import pandas as pd

# Import dataset into program
data = pd.read_csv('Online Retail.csv', delimiter=',')

# Removing rows with NaN values, list wise deletion
data.dropna(axis=0, inplace=True)

# This functionality enables us to filter rows on specified conditions, removing rows not fit for analytics
# Another for of list wise deletion
data = data[(data['Quantity'] > 0)]
data = data[(data['UnitPrice'] > 0)]

# Makes date more readable and program friendly
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate)

# Creating new columns to apply functions on, needed for grouping step
data['Total Revenue'] = data['Quantity'] * data['UnitPrice']
data['Number Of Purchases'] = data['Quantity']
data['Days From Last Purchase'] = data['InvoiceDate']
data['Days From First Purchase'] = data['InvoiceDate']

# Find most recently recorded date in dataframe
column = data["Days From Last Purchase"]
max_value = column.max()

# Subtract each each date recorded from the most recent date recorded, to find number of days elapsed since then
data['Days From Last Purchase'] = max_value - data['Days From Last Purchase']
data['Days From First Purchase'] = max_value - data['Days From First Purchase']

# For each customer, we get their unique statistics such as total revenue spent, total number of purchases,
# most date of most recent purchase, and date of earliest purchase
data = data.groupby(['CustomerID']).agg({'Number Of Purchases': ['sum'], 'Days From Last Purchase': ['min'],
                                         'Days From First Purchase': ['max'], 'Total Revenue': ['sum']})

# Randomizes rows so that they are not ordered by the customer ID numbers
data = data.sample(frac=1).reset_index(drop=False)

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=True)
