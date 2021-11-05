from netpyne import sim
import numpy as np
import matplotlib.pyplot as plt 
from neuron import h

## Load sim from .pkl file 
fn = '../data/v34_batch27_QD_rec/v34_batch27_QD_rec_data.pkl' #'../data/v32_batch28/v32_batch28_data.pkl' #'../data/v34_batch27_QD_rec/v34_batch27_QD_rec_data.pkl'	#'../data/lfpSimFiles/A1_v34_batch27_v34_batch27_0_0.pkl'

sim.load(fn,instantiate=True) # fn should be .pkl netpyne sim file 
# NOTE: instantiate=False makes sim.net.compartCells an empty list 


## Plot LFP 
#lfp_data = np.array(sim.allSimData['LFP']) # LFP data from sim
#sim.analysis.plotLFP(plots=['timeSeries'], timeRange=[8000,8500], electrodes=[5, 10])   # 'PSD'  # , 'spectrogram'


## Try re-creating the lfp plotting lines to see where things get weird 
timeRange = [0,sim.cfg.duration] #[8250,8260] #[0,sim.cfg.duration] # can adjust as desired --> e.g. timeRange = [8000,8500]
#timeRange=[8000,8500]

lfp = np.array(sim.allSimData['LFP'])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)

elec = 2    # this is the electrode you want to plot 

lfpPlot = lfp[:, elec] 


#####
allData = sim.allSimData



# ## Figure out which indices lfp goes to zero to see how often this happens 
lfpZeroInd = np.where(lfpPlot==0)
lfpZeroInd = list(lfpZeroInd[0])

print('lfpZeroInd[0]: ' + str(lfpZeroInd[0]))
print('lfpZeroInd[1]: ' + str(lfpZeroInd[1]))
print('lfpZeroInd[2]: ' + str(lfpZeroInd[2]))
print('lfpZeroInd[3]: ' + str(lfpZeroInd[3]))
print('lfpZeroInd[4]: ' + str(lfpZeroInd[4]))
print('lfpZeroInd[5]: ' + str(lfpZeroInd[5]))
print('lfpZeroInd[6]: ' + str(lfpZeroInd[6]))
print('lfpZeroInd[7]: ' + str(lfpZeroInd[7]))



cfg.recordTraces = {'V_soma':{'sec':'soma', 'loc':0.5, 'var':'v'}, 'I_memb':{'sec':'soma', 'loc':0.5, 'var': 'i'}


#### Plot LFP 
# #plt.plot(t[0:len(lfpPlot)], lfpPlot, linewidth=1.0)
# #plt.plot(t,lfp[:,elec]) # 10
# plt.plot(t, lfpPlot, linewidth=1.0)
# plt.show()


## Testing tr and im 
cell0 = sim.net.compartCells[0]
cell1 = sim.net.compartCells[1]

gid0 = cell0.gid
gid1 = cell1.gid

im0 = cell0.getImemb()
im1 = cell1.getImemb()

print('im0: ' + str(im0))
print('im1: ' + str(im1))

#### look at all membrane currents at the end
count = 0
for i in range(len(sim.net.compartCells)):
	cell = sim.net.compartCells[i]
	im = cell.getImemb()
	if not list(im ==0):
		count += 1

print('count: ' + str(count))

# tr0 = sim.net.recXElectrode.getTransferResistance(gid0)
# tr1 = sim.net.recXElectrode.getTransferResistance(gid1)

# print('tr0: ' + str(tr0))
# print('tr1: ' + str(tr1))

# ecp0 = np.dot(tr0, im0)
# ecp1 = np.dot(tr1, im1)

# print('ecp0: ' + str(ecp0))
# print('ecp1: ' + str(ecp1))


###########################################
# # Loading individual LFP traces
# LFPCellDict = sim.allSimData['LFPCells']
# print('Dict keys for LFPCells: ' + str(LFPCellDict.keys()))

# LFPCells = LFPCellDict.keys()
# for cell in LFPCells:
# 	elec = 4  	# arbitrary -- which electrode do you want to plot?
# 	LFPtrace = LFPCellDict[cell][:,elec]
# 	LFPtrace = list(LFPtrace) ## necessary?
# 	plt.plot(t,LFPtrace)
# 	plt.show()
#### COMMENTED OUT JUST NOW WHEN LOOKING AT MEMBRANE VOLTAGES


###########
## Look at membrane voltages 

allData = sim.allSimData 
print(allData.keys())
# membraneVoltage = allData['V_soma']
# cells = list(membraneVoltage.keys())

# #membraneVoltage[cells[0]]

# plt.plot(t, membraneVoltage[cells[0]])
# plt.show()

###########################################
# ## from cfg json file 
# "electrodes": [
#                     10
#                 ],
#                 "figSize": [
#                     8,
#                     4
#                 ],
#                 "maxFreq": 80,
#                 "plots": [
#                     "timeSeries",
#                     "PSD",
#                     "spectrogram"
#                 ],
#                 "saveData": false,
#                 "saveFig": true,
#                 "showFig": false,
#                 "timeRange": [
#                     1500,
#                     11500
#                 ]


# ## CSD plotting lines from a1dat load.py, in case helpful 
# sampr = 1./dt  # sampling rate (Hz) (1/sec)
# spacing_um =  sim.cfg.recordLFP[1][1] - sim.cfg.recordLFP[0][1] # spacing between electrodes in microns
# # get CSD data 
# CSD = getCSD(lfps=lfp_data,sampr=sampr,spacing_um=spacing_um,vaknin=False) # getCSD() in nhpdat.py 
# # Get tt  -- default unit: MILLISECONDS  -- conversion required -- not like macaque data where default unit is sec
# fullTimeRange = [0,(sim.cfg.duration/1000.0)] # units: seconds   #[0,(sim.cfg.duration/1000.0)] # sim.cfg.duration is a float # timeRange should be the entire sim duration 
# tt = np.arange(fullTimeRange[0],fullTimeRange[1],dt)  # units: seconds 
# dat = CSD