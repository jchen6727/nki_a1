from netpyne import sim 
from netpyne.analysis.tools import *
import matplotlib.pyplot as plt 
import numpy as np

basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/REDO_BBN_CINECA_v36_5656BF_624SOA/'

# filename
filename = 'REDO_BBN_CINECA_v36_5656BF_624SOA_0_0_0_data.pkl'

fn = basedir + filename



sim.load(fn, instantiate=False)


# Get stimulus times 
stimTimes = sim.cfg.ICThalInput['startTime']

# Plot histogram 
## sim.plotting.plotSpikeHist(include=['IC', 'IT2'], timeRange = [1000,10000], showFig=1)


## CALCULATE IC SPIKE TIMES FOR BAR PLOT 
cells_IC, cellGids_IC, netStimLabels_IC = getInclude(['IC'])	# getInclude(include)

IC_stimTimeRanges = {}
for stimTime in stimTimes:
	stimTimeKey = 'stim_' + str(int(stimTime)) + 'ms'
	IC_stimTimeRanges[stimTimeKey] = [stimTime, stimTime + 50]


spkts_IC = {}
for i in IC_stimTimeRanges.keys():
	timeRange = IC_stimTimeRanges[i]
	sel, spkts, spkgids = getSpktSpkid(cellGids_IC, timeRange=timeRange) # using [] is faster for all cells
	spkts_IC[i] = {'sel': sel, 'spkts': spkts, 'spkgids': spkgids, 'timeRange': timeRange}


numStims = len(IC_stimTimeRanges.keys())
x = list(np.arange(1,numStims+1,1))

IC_spkNumsList = []
for q in spkts_IC.keys():
	IC_spkNumsList.append(len(spkts_IC[q]['spkts']))



## PLOT IC SPIKE BAR BLOT ON BOTTOM SUBPLOT 
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
ax2.bar(x, IC_spkNumsList)
#ax2.title('IC spikes')
#plt.bar(x, IC_spkNumsList) #, label=stimTimes)
# --> NEXT GET TITLE, LABELS ALL SORTED OUT! 


#plt.show()



### 
ECorticalPops = []
EThalPops = ['TC', 'TCM']
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6']  # all layers

include=['IT2']
cells, cellGids, netStimLabels = getInclude(include)
## 
# timeRange = [1000,10000]
# sel, spkts, spkgids = getSpktSpkid(cellGids=[] if include == ['allCells'] else cellGids, timeRange=timeRange) # using [] is faster for all cells



