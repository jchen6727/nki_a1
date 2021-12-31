from netpyne import sim


pathToFiles = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'

### batch 57 
batch57Files = ['v34_batch57_3_4_data', 'v34_batch57_3_3_data', 'v34_batch57_3_2_data']

### batch 65
batch65Prefix = 'A1_v34_batch65_'
batch65Files = ['v34_batch65_0_0_data', 'v34_batch65_1_1_data', 'v34_batch65_2_2_data']
#batch65Files = ['A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'A1_v34_batch65_v34_batch65_0_0_data.pkl']



### LOAD THE DATA #### 
batch57_data = {}
batch57_LFP = {}
for file in batch57Files:
	fullFile = pathToFiles + file + '.pkl'
	sim.load(fullFile, instantiate=False)
	fileKey = file.split('v34_')[1].split('_data')[0].split('batch57_')[1]
	batch57_data[fileKey] = sim.allSimData
	batch57_LFP[fileKey] = sim.allSimData['LFP']



batch65_data = {}
batch65_LFP = {}
for file in batch65Files:
	fullFile = pathToFiles + batch65Prefix + file + '.pkl'
	sim.load(fullFile, instantiate=False)
	fileKey = file.split('v34_')[1].split('_data')[0].split('batch65_')[1]
	batch65_data[fileKey] = sim.allSimData
	batch65_LFP[fileKey] = sim.allSimData['LFP']


#### NOW COMPARE #### 


