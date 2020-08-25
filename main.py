# Reading an excel file using Python
import xlrd
import numpy
import matplotlib.pyplot as plt
from datetime import datetime

# Give the location of the file
loc = ('Online Retail.xlsx')

# To open Workbook
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# Taking in row by row from excel file
# Each row is an item
row = []
i = 1
while True:
    try:
        row.append(sheet.row_values(i))
        i += 1
    except IndexError:
        break

def purchaseFreq():

    # Searches each item to see who purchased it
    # Creates dictionary of each customer and their number of card swipes
    frequency = {}
    old_invoice = 0
    for i in row:
        new_invoice = i[0]
        if old_invoice != new_invoice:
            if i[6] != '':
                # Leaves out missing values
                # Disregards creating empty ID key named ''
                frequency.setdefault(i[6], 0)
                frequency[i[6]] = frequency[i[6]] + 1
                old_invoice = new_invoice
            else:
                old_invoice = new_invoice
                continue
        else:
            continue

    print('Times each customer swiped their card: ' + str(frequency))

    # Frequency of each number of card swipes
    # example: there were 5 people who card swiped 35 times
    graphData = {}
    for i in frequency.values():
        graphData.setdefault(i, 0)
        graphData[i] = graphData[i] + 1

    # Placing keys and values in separate lists
    numOfPurchases = []
    for i in graphData.keys():
        numOfPurchases.append(i)
    numOfCustomers = []
    for i in graphData.values():
        numOfCustomers.append(i)

    # Graph creation
    plt.bar(numOfPurchases, numpy.sqrt(numOfCustomers), label='sqrt(Number Of Customers)')
    plt.xlabel('Number Of Purchases')
    plt.ylabel('sqrt(Number Of Customers)')
    plt.title('Purchase Frequencies')
    plt.legend()
    plt.show()

    #fig = plt.figure()
    #plt.boxplot(numOfPurchases)
    #plt.show()

def recentPurchase():
    recentDatesPerCustomer = {}
    currentDate = 0
    for i in row:
        if i[6] != '':
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            recentDatesPerCustomer.setdefault(i[6], 0)
            recentDatesPerCustomer[i[6]] = i[4]
            currentDate = i[4]
        else:
            continue

    currentDate = datetime(2011, 12, 9, 12, 50)
    for i in recentDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(recentDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        recentDatesPerCustomer[i] = numOfDays.days

    recentDateFreq = {}
    for i in recentDatesPerCustomer:
        recentDateFreq.setdefault(recentDatesPerCustomer[i], 0)
        recentDateFreq[recentDatesPerCustomer[i]] = recentDateFreq[recentDatesPerCustomer[i]] + 1

    # Placing keys and values in separate lists
    daysSinceLastPurchase = []
    for i in recentDateFreq.keys():
        daysSinceLastPurchase.append(i)
    numOfCustomers = []
    for i in recentDateFreq.values():
        numOfCustomers.append(i)

    # Graph creation
    plt.bar(daysSinceLastPurchase, numOfCustomers, label='Number Of Customers')
    plt.xlabel('Days Since Last Purchase')
    plt.ylabel('Number Of Customers')
    plt.title('Days From Last Purchase')
    plt.legend()
    plt.show()

    #fig = plt.figure()
    #plt.boxplot(daysSinceLastPurchase)
    #plt.show()

def firstPurchase():
    firstDatesPerCustomer = {}
    for i in row:
        if i[6] != '':
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            firstDatesPerCustomer.setdefault(i[6], i[4])
        else:
            continue

    currentDate = datetime(2011, 12, 9, 12, 50)
    for i in firstDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(firstDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        firstDatesPerCustomer[i] = numOfDays.days

    print(firstDatesPerCustomer)

    recentDateFreq = {}
    for i in firstDatesPerCustomer:
        recentDateFreq.setdefault(firstDatesPerCustomer[i], 0)
        recentDateFreq[firstDatesPerCustomer[i]] = recentDateFreq[firstDatesPerCustomer[i]] + 1

    # Placing keys and values in separate lists
    daysSinceFirstPurchase = []
    for i in recentDateFreq.keys():
        daysSinceFirstPurchase.append(i)
    numOfCustomers = []
    for i in recentDateFreq.values():
        numOfCustomers.append(i)

    # Graph creation
    #plt.bar(daysSinceFirstPurchase, numOfCustomers, label='Number Of Customers')
    #plt.xlabel('Days Since First Purchase')
    #plt.ylabel('Number Of Customers')
    #plt.title('Days From First Purchase')
    #plt.legend()
    #plt.show()

    fig = plt.figure()
    plt.boxplot(daysSinceFirstPurchase)
    plt.show()

def totalRevenue():
    listOfTotalRevenue = {}
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
            and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            listOfTotalRevenue.setdefault(i[6], 0)
            addRevenue = i[3]*i[5]
            if addRevenue < 0:
                addRevenue = -1 * addRevenue
            listOfTotalRevenue[i[6]] = listOfTotalRevenue[i[6]] + addRevenue
        else:
            continue

    revenueIndexDict = {}
    counter = 0
    for i in sorted(listOfTotalRevenue.values(), reverse=True):
        revenueIndexDict.setdefault(counter, 0)
        revenueIndexDict[counter] = i
        counter += 1

    revenueIndex = []
    revenueValue = []
    for i in revenueIndexDict:
        revenueIndex.append(i)
        revenueValue.append(revenueIndexDict[i])

    # Graph creation
    plt.bar(revenueIndex, numpy.sqrt(revenueValue), label='sqrt(Revenue in $')
    plt.ylabel('sqrt(Revenue in $)')
    plt.title('All Revenues From Each Customer')
    plt.legend()
    plt.show()

#purchaseFreq()
# Large cluster of customers who have only purchased once.
# Average customer makes ~30 card swipes.
# Few outliers with over 100 card swipes.

#recentPurchase()
# Large cluster of customers who made their last purchase in the last 30 days.
# Average customer has made last purchase in ~175 days.

#firstPurchase()
# Large cluster of customers who made first purchases in the first 100 days.
# Average customer made first purchase 205 days ago.

#totalRevenue()
# Majority of customers spend little money.
# Quite a few outliers where hundreds of thousands of dollars in revenue.

# CONCLUSIONS:
# The average customer makes few purchases a year, but is a returning customer.
