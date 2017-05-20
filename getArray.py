import random
import numpy as np



def getState(datasize):
    arralist = []
    for i in range(datasize):
        ara = np.zeros((7, 7))
        num = random.randrange(1, 50)
        poslist = random.sample(range(49), num)
        for k in range(len(poslist)):
            x = poslist[k] // 7
            y = poslist[k] % 7
            ara[x][y] = random.sample([-1, 1], 1)[0]
        arralist.append(ara)
    return arralist



