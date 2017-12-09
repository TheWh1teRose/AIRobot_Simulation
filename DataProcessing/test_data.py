import cv2
import numpy as np
import pickle
np.set_printoptions(threshold=np.nan)

file = 'D:/Dokumente/Programmieren/RoboPen/UnitySimulation/AIRobot_Simulation/DataProcessing/traindata/pre/test_data.pickle'
data = pickle.load( open(file, "rb"))
data = np.asarray(data[0])
print(data.shape)
new = data[2,...]
print(new.shape)
new = new[:,:,:3]
print()
test = np.array(new, dtype=np.int8)
print(np.array(test, dtype=np.float32))
print(new.shape)
cv2.imshow('image', np.array(new, dtype=np.int8))
cv2.waitKey(0)
cv2.destroyAllWindows()
