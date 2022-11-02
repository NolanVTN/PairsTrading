from yahoo_fin import options
from yahoo_fin import stock_info as si
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import *
import matplotlib.pyplot as plt
import math
import csv
import array as arr

#claculate the stock coorelations
Day = int(input("Enter start day: Ex:8 ")) 
month = int(input("Enter start month: Ex:11 "))
year = int(input("Enter start year: Ex:2021 "))
today = date.today()
past = date(year,month,Day)
#the the time (in days) that we will be looking at
timeFrame = float((today - past).days)

def interval(t):
        if float(t) > 300:
            interval = '1d'
        #elif (float(t) < 100 and float(t) > 200): 
            #interval = '1wk'
        else:
            interval = '1d'
        return interval
        
def twoDataCoorelation(data1, data2):
    dataN1 = data1[['open']]
    dataN2 = data2[['open']]
    L = len(data1)
    meanData1 = dataN1.sum()/L
    meanData2 = dataN2.sum()/L
    data1Hat = dataN1 - meanData1
    
    data2Hat = dataN2 - meanData2
    data1Data2 = np.multiply(data1Hat,data2Hat)
    Data1sqr = np.multiply(data1Hat, data1Hat)
    Data2sqr = np.multiply(data2Hat, data2Hat)
    avgHat = data1Data2.sum()/L
    avgD1H = Data1sqr.sum()/L
    avgD2H = Data2sqr.sum()/L
    denom = avgHat/(math.sqrt(avgD1H*avgD2H))
    return denom

def historicalVolatility(data1):
    data = data1[['open']]
    dataMean = (data.sum()) / len(data1)
    dataMeanSqr = (data - dataMean) * (data - dataMean)
    avgSumDataMS = dataMeanSqr.sum() / len(data1)
    returnVal = math.sqrt(avgSumDataMS)
    return returnVal
 
readData = pd.read_csv('nasdaq2.csv')
tickerData = readData[['Ticker']]

trest = []
for i in range(0,len(readData)):
    if (len(tickerData.loc[i,'Ticker']) > 0):
        trest.append(tickerData.loc[i,'Ticker'])
        
maxSize = 10
HV = [.1 for i in range(0,maxSize)]
Vol = [.1 for i in range(0,maxSize)]
openDataVal = ["" for i in range(0,maxSize)]
stockTicker = ["" for i in range(0,maxSize) ]
trackerVal = 0

for i in range(0,len(trest)):
    trestData = si.get_data(trest[i], start_date = past, end_date = today, index_as_date = True, interval = interval(timeFrame))
    Hv = historicalVolatility(trestData)
    stockvol = trestData.volume
    stockVol = stockvol.sum()/ len(stockvol)
    for j in range(0,maxSize):
        if(int(stockVol) > Vol[j]):
            Vol[j] = stockVol
            openDataVal[j] = trestData
            stockTicker[j] = trest[i]
            break
            

chosenStock1 = ""
stock1Index = 0
chosenStock2 = ""
stock2Index = 0
chosenStockCoor = 0
for i in range(0, maxSize):
    for j in range((i + 1),maxSize):
        coor = twoDataCoorelation(openDataVal[i], openDataVal[j])
        temp = coor.open
        if(temp > chosenStockCoor):
            chosenStock1 = stockTicker[i]
            chosenStock2 = stockTicker[j]
            stock1Index = i
            stock2Index = j
            chosenStockCoor = temp
  
plt_1 = plt.figure(figsize=(6, 9))

print("The coorelation between the stocks is: ", temp)

plotstockData1 = openDataVal[stock1Index].open
PSD1 = plt.plot(plotstockData1,color = 'red')
print( chosenStock1, " is red")

plotstockData2 = openDataVal[stock2Index].open
PSD2 = plt.plot(plotstockData2,color = 'blue')
print(chosenStock2, " is blue")

plt_2 = plt.figure(figsize=(6, 9))

print("The coorelation between the stocks is: ", temp)

theRatioData = openDataVal[stock1Index].open/openDataVal[stock2Index].open
PSD1 = plt.plot(theRatioData,color = 'red')
print( chosenStock1, "/", chosenStock2, " is red.")

meanRatio = theRatioData.sum()/len(theRatioData)
plotstockData1 = openDataVal[stock1Index].open / meanRatio
PSD1 = plt.plot(plotstockData1,color = 'red')
print( chosenStock1, " is red")

plotstockData2 = openDataVal[stock2Index].open #* meanRatio
PSD2 = plt.plot(plotstockData2,color = 'blue')
print(chosenStock2, " is blue")

#theRatioData.index


#using the optimal least squares error model with y = A * x + B
#A = <x^ y^> / < (x^)^2 >
#B = <y> - A * <x>
L =  int(len(theRatioData.index))

#variance of X and Y
sum2X = [0 for i in range(0, L)]
for i in range(1, L):
    sum2X[i] = sum2X[i - 1] + 1
    
sumXY = np.multiply(sum2X, theRatioData).sum()
sumofX = sum(sum2X)
sumY = theRatioData.sum()
sumXSqr = np.multiply(sum2X, sum2X).sum()

mTop = (L * sumXY) - (sumofX * sumY)
mBottom = (L * sumXSqr) - (sumofX * sumofX)

A = mTop / mBottom
B = (sumY - (A * sumofX)) / L

ratioDataCopy = theRatioData

for i in range(0, len(theRatioData)):
    ratioDataCopy[i] = A*i + B

print(A, "A", B, "B", "The length ", maxIndex)

theRatioData = openDataVal[stock1Index].open/openDataVal[stock2Index].open
modelRatio = plt.plot(ratioDataCopy,color = 'blue')
PlotRatio = plt.plot(theRatioData,color = 'red')

#ploting the error
#to find the error take the ratio data and subtract our linear ratio model 
randomCount1 = 0
randomCount2 = 0
for i in range(0 , L):
    if (theRatioData[i]< ratioDataCopy[i]):
        randomCount1 += 1
    else:
        randomCount2 += 1
print(randomCount1, randomCount2)
modelError = theRatioData - ratioDataCopy
absoluteModelError = abs(modelError)

PlotRatioAnalytics = plt.plot(modelError,color = 'blue') 
plto2 = plt.plot(absoluteModelError, color = 'green')

#modelRatio = plt.plot(modelError,color = 'blue')
 
 PlotRatio = plt.plot(absoluteModelError,color = 'red') 
 
 stock1 = chosenStock1
Day = int(input("Enter option expiration day: Ex:8 "))
month = int(input("Enter option expiration month: Ex:11 "))
year = int(input("Enter option expiration year: Ex:2021 "))
today = date.today()
future = date(year,month,Day)
expiry = future
str(future - today)
pd.set_option("display.max_rows", None, "display.max_columns", None)
options.get_options_chain(stock1)
chain = options.get_options_chain(stock1,expiry)
 
