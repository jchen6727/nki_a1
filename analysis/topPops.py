import json
#import pickle
import pickle5 as pickle
import os


###################
#### FUNCTIONS ####
###################

##################################################
def topIndividualPops(topPopsData, freqBand, region):
	## OUTPUT: bar graph with the top represented population(s) for given osc event subtype
	#### topPopsData	: dict 
	#### freqBand 		: str
	#### region 		: str 

	eventIndices = list(topPopsData[freqBand][region].keys())

	popCounts = {}

	for eventIdx in eventIndices:
		topPops = topPopsData[freqBand][region][eventIdx]
		for pop in topPops:
			if pop in popCounts.keys():
				popCounts[pop] += 1 
			else:
				popCounts[pop] = 1



	popCountsSorted = {k: v for k, v in sorted(popCounts.items(), key=lambda item: item[1], reverse=True)}


	return popCountsSorted





#############################################
def topPopGroups(topPopsData, freqBand, region):
	## OUTPUT: bar graph with the most popular trios for given osc event subtype
	#### topPopsData	: dict 
	#### freqBand 		: str
	#### region 		: str 

	eventIndices = list(topPopsData[freqBand][region].keys())

	trioCounts = {}

	for eventIdx in eventIndices:
		trioSorted = sorted(topPopsData[freqBand][region][eventIdx])
		if str(trioSorted) not in trioCounts.keys():
			trioCounts[str(trioSorted)] = 1
		else:
			trioCounts[str(trioSorted)] += 1

	trioCountsSorted = {k: v for k, v in sorted(trioCounts.items(), key=lambda item: item[1], reverse=True)}


	return trioCountsSorted



################################################
# def popSourceSink(oscEventData,freqBand, region):
	## OUTPUT: Identify / Visualize the sources and sinks for given osc event subtype

	### oscEventData[freqBand][region][subject][eventIdx]['maxPops_avgCSD']['elec'][pop] = CSD value





################################################
############# MAIN CODE BLOCK ##################
################################################

based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/topPops/ECortPops/'


os.chdir(based)

allFiles = os.listdir()

topPopsFiles = []
oscEventFiles = []

for file in allFiles:
	if 'topPops' in file:
		topPopsFiles.append(file)
	if 'oscEventInfo' in file:
		oscEventFiles.append(file)

# print(topPopsFiles)
# print(oscEventFiles)


## LOAD TOP POPS DATA
topPopsData = {}
for topPopFile in topPopsFiles:
	f = open(topPopFile)
	data = json.load(f)
	topPopsData.update(data)

## LOAD OSC EVENT INFO INTO DICT 
oscEventData = {}
for oscEventFile in oscEventFiles:
	with open(oscEventFile, 'rb') as handle:
		data = pickle.load(handle)
	oscEventData.update(data)


### Synthesize findings from topPops Data ###
regions = ['supra', 'gran', 'infra']
freqBands = ['delta', 'theta', 'alpha', 'beta']

ECortPops = ['IT2', 
			 'IT3', 
			 'ITP4', 'ITS4', 
			 'IT5A', 'CT5A', 
			 'IT5B', 'CT5B', 'PT5B', 
			 'IT6', 'CT6']


popCounts = topIndividualPops(topPopsData, 'theta', 'supra')

trioCounts = topPopGroups(topPopsData, 'theta', 'supra')





























