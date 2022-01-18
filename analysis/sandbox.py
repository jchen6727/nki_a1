from netpyne import sim
import numpy as np
#from simDataAnalysis import *



######################## DEBUGGING LFP HEAT PLOTS ########################

waveletInfo = {'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520]},
	'beta': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [456, 572]}, 
	'alpha': {'dataFile': 'A1_v34_batch65_v34_batch65_1_1_data.pkl', 'timeRange': [3111, 3325]}, 
	'theta': {'dataFile': 'A1_v34_batch65_v34_batch65_2_2_data.pkl', 'timeRange': [2785, 3350]}}

freqBand = 'delta'
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/' 

dataFile = based + waveletInfo[freqBand]['dataFile']
timeRange = waveletInfo[freqBand]['timeRange']

print('timeRange = ' + str(timeRange))
print('dataFile = ' + str(dataFile))

sim.load(dataFile, instantiate=False)



### thal pops
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
allPops = list(sim.net.allPops.keys())
pops = [pop for pop in allPops if pop not in thalPops] 			## exclude thal pops 
## output --> >>> pops
# ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6']

### play with lfp data 
timePoint = 20			# pick a timepoint for convenience 

lfpTotal = sim.allSimData['LFP'][timePoint]   # <-- list of 20 (total lfp amplitudes at each electrode for this timepoint)

popLFPLists = []  # <-- list of 36 lists (1 for each pop); each list is length 20 (lfp amplitudes at each electrode for this timepoint)
for pop in pops:
	lfpSublist = sim.allSimData['LFPPops'][pop][timePoint] 
	popLFPLists.append(lfpSublist)
lfpPopTotal = sum(popLFPLists)


### NOW CHECK --> lfpPopTotal == lfpTotal



#### pop-specific 
# testPop = 'NGF1'

# popLFPdata_TOTAL = sim.allSimData['LFPPops'][testPop]
# print('dims of popLFPdata_TOTAL: ' + str(popLFPdata_TOTAL.shape)) 			 # (230000, 20)


# popLFPdata = np.array(sim.allSimData['LFPPops'][testPop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
# print('dims of popLFPdata: ' + str(popLFPdata.shape))
# # dims of new popLFPdata: (20800, 20)

# totalAvg = np.average(popLFPdata)
# print('totalAvg: ' + str(totalAvg))
# totalPeak = np.amax(popLFPdata)
# print('totalPeak: ' + str(totalPeak))


## Checking pop to total comparison
### timepoints should be a for loop I suppose
### lfp totals


#popLFPSum = 0
#for pop in pops:
	#lfpSum = sim.allSimData['LFPPops'][pop][timePoint] #sum(sim.allSimData['LFPPops'][pop][0]) 	# np.sum(sim.allSimData['LFPPops'][pop][0,:])
	#print('lfp amplitude over all electrodes for ' + pop + ' at timepoint ' + str(0) + ': ' + str(lfpSum))
	#popLFPSum+=lfpSum

# print(str(sum(lfpTotal[0])))  #[0])))
# print(str(totalLFPSum))

#lfpTotal[0] == totalLFPSum



### calculate avg and peak 
# for pop in pops:
# 	lfpPopData[pop] = {}

# 	popLFPdata = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]

# 	lfpPopData[pop]['totalLFP'] = popLFPdata
# 	lfpPopData[pop]['totalAvg'] = np.average(popLFPdata)
# 	lfpPopData[pop]['totalPeak'] = np.amax(popLFPdata)


# 	for elec in evalElecs:
# 		elecKey = 'elec' + str(elec)
# 		lfpPopData[pop][elecKey] = {}
# 		lfpPopData[pop][elecKey]['LFP'] = popLFPdata[:, elec]
# 		lfpPopData[pop][elecKey]['avg'] = np.average(popLFPdata[:, elec])
# 		lfpPopData[pop][elecKey]['peak'] = np.amax(popLFPdata[:, elec])




###############################################################
# pathToFiles = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'

# ### batch 57 
# batch57Files = ['v34_batch57_3_4_data', 'v34_batch57_3_3_data', 'v34_batch57_3_2_data']

# ### batch 65
# batch65Prefix = 'A1_v34_batch65_'
# batch65Files = ['v34_batch65_0_0_data', 'v34_batch65_1_1_data', 'v34_batch65_2_2_data']
# #batch65Files = ['A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl']



# ### LOAD THE DATA #### 
# batch57_data = {}
# batch57_LFP = {}
# for file in batch57Files:
# 	fullFile = pathToFiles + file + '.pkl'
# 	sim.load(fullFile, instantiate=False)
# 	fileKey = file.split('v34_')[1].split('_data')[0].split('batch57_')[1]
# 	batch57_data[fileKey] = sim.allSimData
# 	batch57_LFP[fileKey] = sim.allSimData['LFP']



# batch65_data = {}
# batch65_LFP = {}
# for file in batch65Files:
# 	fullFile = pathToFiles + batch65Prefix + file + '.pkl'
# 	sim.load(fullFile, instantiate=False)
# 	fileKey = file.split('v34_')[1].split('_data')[0].split('batch65_')[1]
# 	batch65_data[fileKey] = sim.allSimData
# 	batch65_LFP[fileKey] = sim.allSimData['LFP']



#### NOW COMPARE LFP #### 

##### THAL POP DEBUGGING #####
# pops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
# thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']

# pops_new = [pop for pop in pops if pop not in thalPops]

# for pop in pops:
# 	if any(elem == pop for elem in thalPops):
# 		print('redundant pop: ' + pop)
		#pops.remove(pop)


