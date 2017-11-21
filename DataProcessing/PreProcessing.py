import numpy as np 
import threading
import glob
import sys
import gc


gc.enable()

maxMemoryUsage = 1000
path = 'traindata/*.npy'
file = glob.glob(path)
data = np.array((np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)))



for f in file:
	lastData = np.load(f)
	data = np.vstack((data, lastData))
	if sys.getsizeof(data)>maxMemoryUsage:
		print("test")
		data = np.array((np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)))
		gc.collect()
	#drop 0:0:0:0

