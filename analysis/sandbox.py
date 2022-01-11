# from netpyne import sim


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
pops = ['NGF1', 'IT2', 'SOM2', 'PV2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'SOM4', 'PV4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'SOM5A', 'PV5A', 'VIP5A', 'NGF5A', 'IT5B', 'CT5B', 'PT5B', 'SOM5B', 'PV5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'SOM6', 'PV6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']

pops_new = [pop for pop in pops if pop not in thalPops]

for pop in pops:
	if any(elem == pop for elem in thalPops):
		print('redundant pop: ' + pop)
		#pops.remove(pop)

# for pop in pops:
# 	print(pop)
# 	if pop in pops and pop in thalPops:
# 		print('removing pop: ' + pop)
# 		pops.remove(pop)

# for pop in pops:
# 	if pop in thalPops:
# 		print(pop)
# 		pops.remove(pop)
# 		print('removing pop: ' + str(pop))


# for pop in pops:
# 	print(pop)
	# for thalPop in thalPops:
	# 	print(thalPop)
	# 	if pop == thalPop:
	# 		pops.remove(pop)
	# 		print('removed from pops list: ' + str(pop))


