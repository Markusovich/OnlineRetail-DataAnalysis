# Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Import dataset into program
data = pd.read_csv('Online Retail.csv', delimiter=',')

# Removing rows with NaN values, list wise deletion
data.dropna(axis=0, inplace=True)

# This functionality enables us to filter rows on specified conditions, removing rows not fit for analytics
# Another for of list wise deletion
data = data[(data['Quantity'] > 0)]
data = data[(data['UnitPrice'] > 0)]

# Makes date more readable and program friendly
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate).dt.date

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

# Preforming manipulations on columns, for simplicity purposes
data['Total Revenue'] = data['Total Revenue'].astype(np.int64)
data['Number Of Purchases'] = data['Number Of Purchases'].astype(np.int64)
data['Days From Last Purchase'] = data['Days From Last Purchase'] / np.timedelta64(1, 'D')
data['Days From First Purchase'] = data['Days From First Purchase'] / np.timedelta64(1, 'D')

# Feature engineering, making graphs visualization easier to see
data['sqrt(Total Revenue)'] = np.sqrt(data['Total Revenue'])
data['sqrt(Number Of Purchases)'] = np.sqrt(data['Number Of Purchases'])

# Randomizes rows so that they are not ordered by the customer ID numbers
data = data.sample(frac=1).reset_index(drop=False)

# Removes row between headers and data
data.columns = data.columns.droplevel(1)

# Define number of clusters we want
# We will find the number of clusters specified in a 4d space (4 variables)
Kmean = KMeans(n_clusters=3)
# Cluster based on the select columns
Kmean.fit(data[['sqrt(Number Of Purchases)', 'Days From Last Purchase', 'Days From First Purchase', 'sqrt(Total Revenue)']])

# Get centers for each cluster
print('Mean of each cluster:')
# This algorithm orders the clusters in such fashion that they are always in the same order every run
for i in range(len(Kmean.cluster_centers_)):
    min_idx = i
    for j in range(i + 1, len(Kmean.cluster_centers_)):
        if Kmean.cluster_centers_[min_idx][3] > Kmean.cluster_centers_[j][3]:
            min_idx = j
    Kmean.cluster_centers_[i][3], Kmean.cluster_centers_[min_idx][3] = Kmean.cluster_centers_[min_idx][3], \
                                                                 Kmean.cluster_centers_[i][3]
# Printing the array of each mean of each cluster
print(Kmean.cluster_centers_)
# Cluster 0: New customers
# Cluster 1: Past customers
# Cluster 2: Loyal customers

# Find which cluster each customer belongs to based off select columns
data['Cluster Category'] = pd.Series(Kmean.predict(data[['sqrt(Number Of Purchases)',
                                                         'Days From Last Purchase',
                                                         'Days From First Purchase',
                                                         'sqrt(Total Revenue)']]._get_numeric_data().dropna(axis=1)),
                                     index=data.index)

data = data[['CustomerID',
             'sqrt(Number Of Purchases)',
             'Days From Last Purchase',
             'Days From First Purchase',
             'sqrt(Total Revenue)',
             'Cluster Category']]

# Profile Report
prof = ProfileReport(data)
prof.to_file(output_file='output.html')

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=True)

# ############################## #
# determine k using elbow method #
# ############################## #
# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data[['sqrt(Number Of Purchases)', 'Days From Last Purchase',
                                                'Days From First Purchase', 'sqrt(Total Revenue)']])
    kmeanModel.fit(data[['sqrt(Number Of Purchases)', 'Days From Last Purchase', 'Days From First Purchase',
                         'sqrt(Total Revenue)']])
    distortions.append(sum(np.min(cdist(data[['sqrt(Number Of Purchases)', 'Days From Last Purchase',
                                              'Days From First Purchase', 'sqrt(Total Revenue)']],
                                        kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
