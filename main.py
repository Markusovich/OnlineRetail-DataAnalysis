# Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *

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

# Brings outliers in range, so clustering means have a smaller disposition
data['log(Total Revenue)'] = np.log(data['Total Revenue'])
data['log(Number Of Purchases)'] = np.log(data['Number Of Purchases'])

# Define number of clusters we want
# We will find the number of clusters specified in a 4d space (4 variables)
Kmean = KMeans(n_clusters=5)
# Cluster based on the select columns
# fit function finds the clusters based on the dataframe argument
# Here, I disregarded customer id as one of the variables to be considered,
# because their value does not have a known connection to the customers shopping habits.
Kmean.fit(data[['log(Number Of Purchases)', 'Days From Last Purchase', 'Days From First Purchase', 'log(Total Revenue)']])

# Printing the array of each mean of each cluster
i = 0
for i in range(Kmean.n_clusters):
    print('Cluster ' + str(i) + ': ')
    print('log(Number Of Purchases): ' + str(Kmean.cluster_centers_[i][0]))
    print('Days From Last Purchase: ' + str(Kmean.cluster_centers_[i][1]))
    print('Days From First Purchase: ' + str(Kmean.cluster_centers_[i][2]))
    print('log(Total Revenue): ' + str(Kmean.cluster_centers_[i][3]))

# Clusters are: Highest spenders/most loyal customers, very new customers, infrequent customers,
# high spenders/mostly loyal customers, and former customers/lowest spenders.
print('')
print('Clusters are: Highest spenders/most loyal customers, very new customers, infrequent customers '
      'high spenders/mostly loyal customers, and former customers/lowest spenders.')
print('')

# Find which cluster each customer belongs to based off select columns
# Again, not including the customer id column because its numerical value has no say in the customers habits
# Creates new column for each customer noting the cluster they belong to.
# Establishing the training set
data['Cluster Category'] = pd.Series(Kmean.predict(data[['log(Number Of Purchases)',
                                                         'Days From Last Purchase',
                                                         'Days From First Purchase',
                                                         'log(Total Revenue)']]._get_numeric_data().dropna(axis=1)),
                                     index=data.index)

# Reordering of the columns for display
data = data[['CustomerID',
             'log(Number Of Purchases)',
             'Days From Last Purchase',
             'Days From First Purchase',
             'log(Total Revenue)',
             'Cluster Category']]

# Splitting dataset
# X_train and y_train will make a pattern where the algorithm can guess what cluster a customer from the testing
# set belongs to.
# Again, left out customer id because it does not play a factor in finding the cluster
X_train, X_test, y_train, y_test = \
    train_test_split(data[['log(Number Of Purchases)',
             'Days From Last Purchase',
             'Days From First Purchase',
             'log(Total Revenue)']], data['Cluster Category'], test_size=0.3, random_state=0)

pipelineModel = RandomForestClassifier()
pipelineModel.fit(X_train, y_train)

# The random forest model had the best accuracy out of any other models I tested previously
print('Testing accuracy: ')
print(pipelineModel.score(X_test, y_test))

# Sending model to file so that it can be used elsewhere
from joblib import dump, load
dump(pipelineModel, 'filename.joblib')

# Use this line of code to load back the model
#from joblib import dump, load
#pipeline_randomforest = load('filename.joblib')

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
print('You belong to cluster ' + str(pipelineModel.predict([[np.log(numOfPurchases), daysLast,
                                                                     daysFirst, np.log(totalRev)]])[0]))

# Stores modified dataframe in new file
data.to_csv('Clean Data.csv', index=True)

# Profile Report
prof = ProfileReport(data)
prof.to_file(output_file='output.html')

# Finds optimal k using elbow method.
# ############################## #
# determine k using elbow method #
# ############################## #
# k means determine k
#distortions = []
#K = range(1, 20)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(data[['log(Number Of Purchases)', 'Days From Last Purchase',
#                                                'Days From First Purchase', 'log(Total Revenue)']])
#    kmeanModel.fit(data[['log(Number Of Purchases)', 'Days From Last Purchase', 'Days From First Purchase',
#                         'log(Total Revenue)']])
#    distortions.append(sum(np.min(cdist(data[['log(Number Of Purchases)', 'Days From Last Purchase',
#                                              'Days From First Purchase', 'log(Total Revenue)']],
#                                        kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

# Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()
