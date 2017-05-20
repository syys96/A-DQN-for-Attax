import generateTrainData
import shelve

blackData = generateTrainData.getTraindata(1000,1)
data1 = shelve.open('D:/blackTrainData')
data1['blackData'] = blackData
data1.close()

