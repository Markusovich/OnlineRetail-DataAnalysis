# Libraries
import pandas as pd
import numpy as np
import tf as tf
from pandas_profiling import ProfileReport
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
'''
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

# Removing outliers
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 2.5).all(axis=1)
outlier_entries = (abs_z_scores >= 2.5).all(axis=1)
# Dataframe for outliers
outlier_data = data[outlier_entries]
# Original data without the outliers
data = data[filtered_entries]

# Utilizing pickle code to avoid retraining pipeline every run
from joblib import dump, load
#dump(pipelineClustering, 'pipelineClusteringPickle.joblib')
pipelineClustering = load('pipelineClusteringPickle.joblib')

# Not needed because of pickle
#pipelineClustering = Pipeline([('scalar', MinMaxScaler()), ('kmeans', KMeans(n_clusters=4))])
# Not needed because of pickle
#pipelineClustering.fit_transform(data[['Number Of Purchases', 'Days From Last Purchase', 'Days From First Purchase', 'Total Revenue']])

# This algorithm orders the clusters in such fashion that they are always in the same order every run
for i in range(len(pipelineClustering['kmeans'].cluster_centers_)):
    min_idx = i
    for j in range(i + 1, len(pipelineClustering['kmeans'].cluster_centers_)):
        if pipelineClustering['kmeans'].cluster_centers_[min_idx][3] > pipelineClustering['kmeans'].cluster_centers_[j][3]:
            min_idx = j
    pipelineClustering['kmeans'].cluster_centers_[i][3], pipelineClustering['kmeans'].cluster_centers_[min_idx][3] = pipelineClustering['kmeans'].cluster_centers_[min_idx][3], \
                                                                 pipelineClustering['kmeans'].cluster_centers_[i][3]

# Printing the array of each mean of each cluster
for i in range(pipelineClustering['kmeans'].n_clusters):
    print('Cluster ' + str(i) + ': ')
    print('Number Of Purchases: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][0]))
    print('Days From Last Purchase: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][1]))
    print('Days From First Purchase: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][2]))
    print('Total Revenue: ' + str(pipelineClustering['kmeans'].cluster_centers_[i][3]))

# Find which cluster each customer belongs to in out dataset
# Again, not including the customer id column because its numerical value has no say in the customers habits
# Creates new column for each customer noting the cluster they belong to.
# We now have our dataset where each customer has a column for the cluster they belong to
data['Cluster Category'] = pd.Series(pipelineClustering.fit_predict(data[['Number Of Purchases',
                                                         'Days From Last Purchase',
                                                         'Days From First Purchase',
                                                         'Total Revenue']]._get_numeric_data().dropna(axis=1)),
                                     index=data.index)

data['Cluster Category'].replace({0: 'New Customer', 1: 'Non-frequent Customer',
                                  2: 'Loyal Customer', 3: 'High Spender/Loyal Customer'}, inplace=True)

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

'''
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

app = FastAPI()

clusterDB = []

class Customer(BaseModel):
    name: str
    numOfPurchases: int
    daysLast: int
    daysFirst: int
    totalRev: float

@app.get('/')
def index():
    return {'Go to path': '~/docs'}

@app.post('/customers')
def addCustomer(customer: Customer):
    # Utilizing pickle code to avoid retraining pipeline every run
    # dump(model, 'clusterModel.joblib')
    pipelineClustering = load('pipelineClusteringPickle.joblib')
    clusterDict = {0: 'You belong to the new customer cluster. You most likely show signs of low spending.',
                   1: 'You belong to the non-frequent buyer cluster. You do not have a long record of shopping.',
                   2: 'You belong to the loyal buyer cluster. You may have a long record of purchases.',
                   3: 'You belong to the highest spender cluster. You belong to the category of the highest spenders. '
                      'You also may have a large record, and you shop frequently.'}
    clusterDB.append({'Name': customer.name, 'Category': clusterDict[pipelineClustering.predict([[customer.numOfPurchases, customer.daysLast, customer.daysFirst, customer.totalRev]])[0]]})
    return clusterDB[-1]

@app.get('/customers')
def seeCustomers():
    return clusterDB

@app.get('/customers/{index}')
def getCustomer(index: int):
    return clusterDB[index-1]

'''
# Finds optimal k using elbow method.
# ############################## #
# determine k using elbow method #
# ############################## #
# k means determine k
distortions = []
K = range(1, 9)
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
'''
