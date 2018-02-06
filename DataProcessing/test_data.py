import cv2
import numpy as np
import pickle
import time
#np.set_printoptions(threshold=np.nan)

file = 'F:/Dokumente/Programmieren/RoboPen/UnitySimulation/AIRobot_Simulation/DataProcessing/traindata/pre/data_1_0114180931'
data = pickle.load( open(file, "rb"))
x = data[0]
y = data[1]
for i in range(len(data[0])):
    x_now = x[i]
    y_now = y[i]
    print(y_now)
    cv2.imshow('screen', np.array(x_now[:,:,:3],dtype=np.int8))
    cv2.imshow('posMatrixX', x_now[:,:,3])
    cv2.imshow('posMatrixY', x_now[:,:,4])
    cv2.imshow('posMatrixZ', x_now[:,:,5])
    if cv2.waitKey(25) & 0xFF == ord('q'): #quit statement
        cv2.destroyAllWindows()
        break
    time.sleep(1)
