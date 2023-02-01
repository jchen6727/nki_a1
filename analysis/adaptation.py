from netpyne import sim 
from netpyne.analysis.tools import *


basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/BBN/REDO_BBN_CINECA_v36_5656BF_624SOA/'

# filename
filename = 'REDO_BBN_CINECA_v36_5656BF_624SOA_0_0_0_data.pkl'

fn = basedir + filename



sim.load(fn, instantiate=False)


# Get stimulus times 
stimTimes = sim.cfg.ICThalInput['startTime']

# Plot histogram 
## sim.plotting.plotSpikeHist(include=['IC', 'IT2'], timeRange = [1000,10000], showFig=1)


## OBJECTIVE: CALCULATE NUMBER OF SPIKES IN A GIVEN TIME RANGE
include=['IT2']
cells, cellGids, netStimLabels = getInclude(include)


timeRange = [1000,10000]
sel, spkts, spkgids = getSpktSpkid(cellGids=[] if include == ['allCells'] else cellGids, timeRange=timeRange) # using [] is faster for all cells
