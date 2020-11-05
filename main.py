# Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# Import dataset into program
# dataframe will be designated as 'data'
data = pd.read_csv('Online Retail.csv', delimiter=',')

# Removing rows with NaN values, list wise deletion
# In the near future I may consider a different approach to handling these missing values
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

# Randomizes rows so that they are not ordered by the customer ID numbers
data = data.sample(frac=1).reset_index(drop=False)

# Removes row between headers and data
data.columns = data.columns.droplevel(1)

# Pickled a pipeline where I scale data and then apply kmeans
# Chose MinMaxScaler as the scalar because we do not want the dataset to revolve around outliers
# Example: With Robust scalar, most of the non outliers were placed in the low spending cluster, vs minmax
#scalar had a balanced number of customers in each cluster.
from joblib import dump, load
pipelineClustering = load('filename.joblib')

# Printing the array of each mean of each cluster
for i in range(pipelineClustering['kmeans'].n_clusters):
    print('Cluster ' + str(i) + ': ')
    print('Number Of Purchases: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][0]))
    print('Days From Last Purchase: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][1]))
    print('Days From First Purchase: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][2]))
    print('Total Revenue: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][3]))

# Clusters are: Highest spenders/most loyal customers, new customers, infrequent customers,
# and former customers/lowest spenders.
print('')
print('Clusters are: Highest spenders/most loyal customers, new customers, infrequent customers, '
      'and former customers/lowest spenders.')
print('')

# Find which cluster each customer belongs to in out dataset
# Again, not including the customer id column because its numerical value has no say in the customers habits
# Creates new column for each customer noting the cluster they belong to.
# We now have our dataset where each customer has a column for the cluster they belong to
data['Cluster Category'] = pd.Series(pipelineClustering.predict(data[['Number Of Purchases',
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

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=True)

# Profile Report
prof = ProfileReport(data)
prof.to_file(output_file='output.html')

# User input, assigns them a cluster based on their features
print('Enter customer features')
print("Number Of Purchases: ")
numOfPurchases = int(input())
print("Days From Last Purchase: ")
daysLast = int(input())
print("Days From First Purchase: ")
daysFirst = int(input())
print("Total Revenue: ")
totalRev = int(input())
print('You belong to cluster ' + str(pipelineClustering.predict([[numOfPurchases, daysLast,
                                                                     daysFirst, totalRev]])[0]))

# As a plus, I added some supervised learning
print('Supervised learning model:')
X_train, X_test, y_train, y_test = \
    train_test_split(data[['Number Of Purchases',
             'Days From Last Purchase',
             'Days From First Purchase',
             'Total Revenue']], data['Cluster Category'], test_size=0.3, random_state=0)
# Pipeline
pipeline_randomforest = Pipeline([('scalar', MinMaxScaler()), ('rf_classifier', RandomForestClassifier())])
pipeline_randomforest.fit(X_train, y_train)
# The random forest model had the best accuracy out of any other models I tested previously
# Accuracy is between 98-99%
print('Testing accuracy: ')
print(pipeline_randomforest.score(X_test, y_test))

# Finds optimal k using elbow method.
# ############################## #
# determine k using elbow method #
# ############################## #
# k means determine k
#distortions = []
#K = range(1, 20)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(data[['Number Of Purchases', 'Days From Last Purchase',
#                                                'Days From First Purchase', 'Total Revenue']])
#    kmeanModel.fit(data[['Number Of Purchases', 'Days From Last Purchase', 'Days From First Purchase',
#                         'Total Revenue']])
#    distortions.append(sum(np.min(cdist(data[['Number Of Purchases', 'Days From Last Purchase',
#                                              'Days From First Purchase', 'Total Revenue']],
#                                        kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()
