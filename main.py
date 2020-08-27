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

def univariateAnalysis():
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

    f1 = plt.figure(1)
    plt.title('Number of Purchases')
    plt.boxplot(numOfPurchases)


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

    f2 = plt.figure(2)
    plt.title('Days since last purchase')
    plt.boxplot(daysSinceLastPurchase)


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

    f3 = plt.figure(3)
    plt.title('Days since first purchase')
    plt.boxplot(daysSinceFirstPurchase)


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

    sumOfRevenue = sum(revenueIndexDict.values())
    n = 0
    numOfCustomers = 0
    while n <= 0.05*sumOfRevenue:
        n += revenueIndexDict[numOfCustomers]
        numOfCustomers += 1
    print('Around 5% of the total revenue is spent by the top ' + str(numOfCustomers) + ' customers')
    while n <= 0.1*sumOfRevenue:
        n += revenueIndexDict[numOfCustomers]
        numOfCustomers += 1
    print('Around 10% of the total revenue is spent by the top ' + str(numOfCustomers) + ' customers')
    while n <= 0.25*sumOfRevenue:
        n += revenueIndexDict[numOfCustomers]
        numOfCustomers += 1
    print('Around 25% of the total revenue is spent by the top ' + str(numOfCustomers) + ' customers')
    while n <= 0.5*sumOfRevenue:
        n += revenueIndexDict[numOfCustomers]
        numOfCustomers += 1
    print('Around 50% of the total revenue is spent by the top ' + str(numOfCustomers) + ' customers')
    while n <= 0.75*sumOfRevenue:
        n += revenueIndexDict[numOfCustomers]
        numOfCustomers += 1
    print('Around 75% of the total revenue is spent by the top ' + str(numOfCustomers) + ' customers')

    revenueIndex = []
    revenueValue = []
    for i in revenueIndexDict:
        revenueIndex.append(i)
        revenueValue.append(revenueIndexDict[i])

    print('Mean is ' + str(sumOfRevenue/len(revenueValue)))
    if len(revenueValue) % 2 == 0:
        median1 = revenueValue[len(revenueValue) // 2]
        median2 = revenueValue[len(revenueValue) // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = revenueValue[len(revenueValue) // 2]
    print('Median is ' + str(median))
    print('Since the median is much lower than the mean, this tells me the distribution is skewed right.')

    # Graph creation
    f4 = plt.figure(4)
    plt.bar(revenueIndex, numpy.sqrt(revenueValue), label='sqrt(Revenue in $')
    plt.ylabel('sqrt(Revenue in $)')
    plt.title('All Revenues From Each Customer')
    plt.legend()


def bivariateAnalysis():
    frequency = {}
    old_invoice = 0
    for i in row:
        new_invoice = i[0]
        if old_invoice != new_invoice:
            if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                    and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
                frequency.setdefault(i[6], 0)
                frequency[i[6]] = frequency[i[6]] + 1
                old_invoice = new_invoice
            else:
                old_invoice = new_invoice
                continue
        else:
            continue

    recentDatesPerCustomer = {}
    currentDate = 0
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            recentDatesPerCustomer.setdefault(i[6], 0)
            recentDatesPerCustomer[i[6]] = i[4]
            currentDate = i[4]
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in recentDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(recentDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        recentDatesPerCustomer[i] = numOfDays.days

    f5 = plt.figure(5)
    plt.scatter(numpy.sqrt(list(frequency.values())), numpy.sqrt(list(recentDatesPerCustomer.values())))
    plt.title('Frequency vs Number of days from last purchase')
    plt.xlabel('sqrt(Purchase frequency)')
    plt.ylabel('sqrt(Days since last purchase)')


    frequency = {}
    old_invoice = 0
    for i in row:
        new_invoice = i[0]
        if old_invoice != new_invoice:
            if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                    and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
                frequency.setdefault(i[6], 0)
                frequency[i[6]] = frequency[i[6]] + 1
                old_invoice = new_invoice
            else:
                old_invoice = new_invoice
                continue
        else:
            continue

    firstDatesPerCustomer = {}
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            firstDatesPerCustomer.setdefault(i[6], i[4])
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in firstDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(firstDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        firstDatesPerCustomer[i] = numOfDays.days

    f6 = plt.figure(6)
    plt.scatter(numpy.sqrt(list(frequency.values())), numpy.sqrt(list(firstDatesPerCustomer.values())))
    plt.title('Frequency vs Number of days from first purchase')
    plt.xlabel('sqrt(Purchase frequency)')
    plt.ylabel('sqrt(Days since first purchase)')


    frequency = {}
    old_invoice = 0
    for i in row:
        new_invoice = i[0]
        if old_invoice != new_invoice:
            if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                    and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
                frequency.setdefault(i[6], 0)
                frequency[i[6]] = frequency[i[6]] + 1
                old_invoice = new_invoice
            else:
                old_invoice = new_invoice
                continue
        else:
            continue

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

    f7 = plt.figure(7)
    plt.scatter(numpy.sqrt(list(frequency.values())), numpy.sqrt(list(listOfTotalRevenue.values())))
    plt.title('Frequency of purchases vs Money spent')
    plt.xlabel('sqrt(Purchase frequency)')
    plt.ylabel('sqrt(Total revenue)')


    recentDatesPerCustomer = {}
    currentDate = 0
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            recentDatesPerCustomer.setdefault(i[6], 0)
            recentDatesPerCustomer[i[6]] = i[4]
            currentDate = i[4]
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in recentDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(recentDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        recentDatesPerCustomer[i] = numOfDays.days

    firstDatesPerCustomer = {}
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            firstDatesPerCustomer.setdefault(i[6], i[4])
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in firstDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(firstDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        firstDatesPerCustomer[i] = numOfDays.days

    f8 = plt.figure(8)
    plt.scatter(numpy.sqrt(list(recentDatesPerCustomer.values())), numpy.sqrt(list(firstDatesPerCustomer.values())))
    plt.title('Number of days from last purchase vs Number of days from first purchase')
    plt.xlabel('sqrt(Days since last purchase)')
    plt.ylabel('sqrt(Days since first purchase)')


    recentDatesPerCustomer = {}
    currentDate = 0
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            recentDatesPerCustomer.setdefault(i[6], 0)
            recentDatesPerCustomer[i[6]] = i[4]
            currentDate = i[4]
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in recentDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(recentDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        recentDatesPerCustomer[i] = numOfDays.days

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

    f9 = plt.figure(9)
    plt.scatter(numpy.sqrt(list(recentDatesPerCustomer.values())), numpy.sqrt(list(listOfTotalRevenue.values())))
    plt.title('Number of days from last purchase vs Money spent')
    plt.xlabel('sqrt(Days since last purchase)')
    plt.ylabel('sqrt(Money spent)')


    firstDatesPerCustomer = {}
    for i in row:
        if (i[0] != '' and i[1] != '' and i[2] != '' and i[3] != ''
                and i[4] != '' and i[5] != '' and i[6] != '' and i[7] != ''):
            # Leaves out missing values
            # Disregards creating empty ID key named ''
            firstDatesPerCustomer.setdefault(i[6], i[4])
        else:
            continue
    currentDate = datetime(2011, 12, 10, 12, 50)
    for i in firstDatesPerCustomer:
        realDate = datetime(*xlrd.xldate_as_tuple(firstDatesPerCustomer[i], 0))
        numOfDays = currentDate - realDate
        firstDatesPerCustomer[i] = numOfDays.days

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

    f10 = plt.figure(10)
    plt.scatter(numpy.sqrt(list(firstDatesPerCustomer.values())), numpy.sqrt(list(listOfTotalRevenue.values())))
    plt.title('Number of days from first purchase vs Money spent')
    plt.xlabel('sqrt(Days since first purchase)')
    plt.ylabel('sqrt(Money spent)')
    plt.show()


univariateAnalysis()
bivariateAnalysis()

#Final notes: find best fit line/curve in bivariates
