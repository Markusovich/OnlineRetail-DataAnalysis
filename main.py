import pandas as pd
import datetime as dt

# Import dataset into program
data = pd.read_csv('Online Retail.csv', delimiter=',')

# Removes a column from the dataset, or multiple columns
removed_list = ['InvoiceNo', 'StockCode', 'Country']
data.drop(removed_list, axis=1, inplace=True)

# Removing rows with NaN values
data.dropna(axis=0, inplace=True)

# This functionality enables us to filter rows on specified conditions, removing rows not fit for analytics
data = data[(data['Quantity'] > 0)]
data = data[(data['UnitPrice'] > 0)]

# Reordering data columns, for easier reading
data = data[['CustomerID', 'Description', 'Quantity', 'UnitPrice', 'InvoiceDate']]

# We can tell how many unique customers have purchased from us
# print(str(len(data['CustomerID'].unique())) + ' Customers')

data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate)

print(data.sample(20))

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=False)
