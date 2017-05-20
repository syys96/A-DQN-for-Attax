import generateTrainData
import network
import shelve
datasize = 1000
blackNet = network.Network([49, 100, 200, 792])
blacktraindata = generateTrainData.getTraindata(datasize,1)
for i in range(1):
    blackData = shelve.open('E:/blackData')
    try:
        blackNet.weights = blackData['weights']
        blackNet.biases = blackData['biaes']
    except:
        pass
    blackNet.SGD(blacktraindata, 30, 10, 0.9, test_data=None)
    blackData['weights'] = blackNet.weights
    blackData['biaes'] = blackNet.biases
    blackData.close()
    print(i+1,' times training over!')




