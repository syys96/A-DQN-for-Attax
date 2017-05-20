import alphabe
import getArray
import getposlist
import numpy as np

def getTraindata(datasize,color):
    trainData = []
    inputArrays = getArray.getState(datasize)
    action_dict = getposlist.action_pos()
    action_num = len(action_dict)
    for state in inputArrays:
        action = alphabe.search(state,1,-50,50,color)
        pos = getposlist.getaction_pos(action,action_dict)
        outputArray = np.zeros((action_num,1))
        outputArray[pos][0] = 1.0
        trainData.append((state.reshape((49,1)),outputArray))
    return trainData

