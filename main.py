# Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Import dataset into program
# dataframe will be designated as 'data'
data = pd.read_csv('Online Retail.csv', delimiter=',')

# Removing rows with NaN values, list wise deletion
# In the near future I may consider a different approach to handling these missing values
data.dropna(axis=0, inplace=True)

# Makes date more readable and program friendly
data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate).dt.date

# Creating new columns to apply functions on, needed for grouping step
data['Total Revenue'] = data['Quantity'] * data['UnitPrice']
data['Number Of Purchases'] = data['Quantity']
data['Days From Last Purchase'] = data['InvoiceDate']
data['Days From First Purchase'] = data['InvoiceDate']

# Find most recently recorded date in dataframe
# This will be needed so that we can subtract a given date from it
column = data["Days From Last Purchase"]
max_value = column.max()

# Subtract each each date recorded from the most recent date recorded, to find number of days elapsed since then
data['Days From Last Purchase'] = max_value - data['Days From Last Purchase']
data['Days From First Purchase'] = max_value - data['Days From First Purchase']

# For each customer, we get their unique statistics such as total revenue spent, total number of purchases,
# most date of most recent purchase, and date of earliest purchase
# This code makes a unique table for each customer id, finds the sum of all of their purchases, min days
# from last purchase, max days from first purchase, and sum of all the revenue.
# Thus, for each customer, their own unique row is created containing the features listed below.
data = data.groupby(['CustomerID']).agg({'Number Of Purchases': ['sum'], 'Days From Last Purchase': ['min'],
                                         'Days From First Purchase': ['max'], 'Total Revenue': ['sum']})

# Preforming manipulations on columns, for simplicity purposes
# Making features easier to read
data['Total Revenue'] = data['Total Revenue'].astype(np.int64)
data['Number Of Purchases'] = data['Number Of Purchases'].astype(np.int64)
data['Days From Last Purchase'] = data['Days From Last Purchase'] / np.timedelta64(1, 'D')
data['Days From First Purchase'] = data['Days From First Purchase'] / np.timedelta64(1, 'D')

# This code removes outliers that are 3 standard deviations away from the mean
# z score is a value assigned to all data points to measure its distance from the mean
# If the absolute value of the z score is 3 or greater, it is disregarded
# I chose to implement this code because the extreme outliers were making my cluster means unrealistic
# If I chose to keep outliers, I would have to consider 10 or more clusters in order to accurately group
# all the data points. It would be a pain to find a unique name and description for that many clusters.
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

# Randomizes rows so that they are not ordered by the customer ID numbers
data = data.sample(frac=1).reset_index(drop=False)

# Removes row between headers and data
data.columns = data.columns.droplevel(1)

# Define number of clusters we want
# We will find the number of clusters specified in a 4d space (4 variables)
Kmean = KMeans(n_clusters=4)
# Cluster based on the select columns
# fit function finds the clusters based on the dataframe argument
# Here, I disregarded customer id as one of the variables to be considered,
# because their value does not have a known connection to the customers shopping habits.
Kmean.fit(data[['Number Of Purchases', 'Days From Last Purchase', 'Days From First Purchase', 'Total Revenue']])

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
# Cluster 0: Low spenders
# Cluster 1: Most loyal customers, 2nd lowest spenders
# Cluster 2: Above average high revenue
# Cluster 3: Highest spenders, frequent customers

# Find which cluster each customer belongs to based off select columns
# Again, not including the customer id column because its numerical value has no say in the customers habits
# Creates new column for each customer noting the cluster they belong to.
data['Cluster Category'] = pd.Series(Kmean.predict(data[['Number Of Purchases',
                                                         'Days From Last Purchase',
                                                         'Days From First Purchase',
                                                         'Total Revenue']]._get_numeric_data().dropna(axis=1)),
                                     index=data.index)

# Reordering of the columns for display
data = data[['CustomerID',
             'Number Of Purchases',
             'Days From Last Purchase',
             'Days From First Purchase',
             'Total Revenue',
             'Cluster Category']]

# Removes rows with negative stats
# Does not seem normal for a customer to have a net negative number of purchases
data = data[(data['Number Of Purchases'] > 0)]
data = data[(data['Total Revenue'] > 0)]

# Profile Report
prof = ProfileReport(data)
prof.to_file(output_file='output.html')

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=True)

# Code I stole. Finds optimal k using elbow method.
# ############################## #
# determine k using elbow method #
# ############################## #
# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(data[['Number Of Purchases', 'Days From Last Purchase',
                                                'Days From First Purchase', 'Total Revenue']])
    kmeanModel.fit(data[['Number Of Purchases', 'Days From Last Purchase', 'Days From First Purchase',
                         'Total Revenue']])
    distortions.append(sum(np.min(cdist(data[['Number Of Purchases', 'Days From Last Purchase',
                                              'Days From First Purchase', 'Total Revenue']],
                                        kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
