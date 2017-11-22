import numpy as np
import glob
import sys
import gc
import time
import datetime
import pandas as pd
import preprocessingUtils as pu

gc.enable()

maxMemoryUsage = 1000
path = 'traindata/raw/*.npy'
file = glob.glob(path)
data = np.array([np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)])

def saveData(data):
	#delete the first placeholder column
	data = np.delete(data, 0, 0)
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')
	np.save("traindata/pre/data_1_"+st ,data)
	print('saved data to disk')

for f in file:
	lastoldData = np.load(f)
	data = np.vstack((data, np.delete(lastoldData, 1, 0)))
	print("load: " + f)
	if sys.getsizeof(data)>maxMemoryUsage:
		data = pu.dropData(np.array([0,0,0,0,0,0,0,0]), data)
		print("Drop")
		data = np.delete(data, [0,1,2], 0)
		data = pu.dropData(np.array([0,0,0,0,0,0,1,0]), data)
		print("Drop")
		data = np.delete(data, [0,1,2], 0)
		data = pu.dropData(np.array([0,0,0,0,0,0,0,1]), data)
		print("Drop")
		data = np.delete(data, [0,1,2], 0)
		gc.collect()
		data = np.random.shuffle(data)
		pu.qualifyData(data)
		saveData(data)
		data = np.array([np.array((300,300,6), dtype=np.float64), np.array(8, dtype=np.int8)])
		gc.collect()


data = pu.dropData(np.array([0,0,0,0,0,0,0,0]), data)
print("Drop")
gc.collect()
pu.qualifyData(data)
saveData(data)