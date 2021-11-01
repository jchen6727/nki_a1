"""
plotPaper.py

Wavelet and CSD/LFP plotting for oscillation events in A1 model, using NetPyNE 

Contributors: ericaygriffith@gmail.com 
"""
from netpyne import sim
import os
import utils
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import netpyne
from numbers import Number


### set layer bounds:
layerBounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

### Cell populations: 
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']#, 'IC']
L1pops = ['NGF1']
L2pops = ['IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2']
L3pops = ['IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3']
L4pops = ['ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4']
L5Apops = ['IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A']
L5Bpops = ['IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B']
L6pops = ['IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']
ECortPops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B']
ICortPops = ['NGF1', 
			'PV2', 'SOM2', 'VIP2', 'NGF2', 
			'PV3', 'SOM3', 'VIP3', 'NGF3',
			'PV4', 'SOM4', 'VIP4', 'NGF4',
			'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
			'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
			'PV6', 'SOM6', 'VIP6', 'NGF6']
EThalPops = ['TC', 'TCM', 'HTC']				# TEpops = ['TC', 'TCM', 'HTC']
IThalPops = ['IRE', 'IREM', 'TI', 'TIM']		# TIpops = ['IRE', 'IREM', 'TI', 'TIM']
reticPops = ['IRE', 'IREM']
matrixPops = ['TCM', 'TIM', 'IREM']
corePops = ['TC', 'HTC', 'TI', 'IRE']


### for COLORS in LFP PLOTTING !! 
colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
            [0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
            [0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
            [0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3


### set path to data files
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/spont/'
#based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/v34_batch27_0_3_NGF1_IT5A_CT5A_SHORT/'	#v34_batch27_0_3_IT2_PT5B_SHORT/'
based = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/shortRuns/'

### get .pkl data filenames 
allFiles = os.listdir(based)
allDataFiles = []
for file in allFiles:
	if '.pkl' in file:
		allDataFiles.append(file)

testFiles = [] #['v34_batch27_0_3_NGF1_IT5A_CT5A_SHORT_data.pkl']#['v34_batch27_0_3_IT2_PT5B_SHORT_data.pkl'] #['A1_v34_batch27_v34_batch27_0_3.pkl']


###### Set timeRange ######

timeRange = [175, 350] 			## !! 0_3 SHORT RUN 



###### LOADING SIM DATA and GETTING LFP CONTRIBUTION DATA ####
if len(testFiles) > 0:
	dataFiles = testFiles
else:
	dataFiles = allDataFiles 



allLFPData = {}
allLFPData['LFPPops'] = {}

for fn in dataFiles:
	fullPath = based + fn
	sim.load(fullPath, instantiate=False)

	# Create time lists 
	fullTimeRange = [0, sim.cfg.duration]
	t_full = np.arange(fullTimeRange[0], fullTimeRange[1], sim.cfg.recordStep)
	t_full = list(t_full)  # turn into a list so .index function can be used 

	# NOTE: timeRange is declared earlier
	t = np.arange(timeRange[0], timeRange[1], sim.cfg.recordStep)  ## make an array w/ these time points
	t = list(t)

	# Find the indices of the timeRange within the full range, to correspond to the desired segment of LFP data
	beginIndex = t_full.index(timeRange[0])
	endIndex = t_full.index(timeRange[-1])


	## LFP contributions knitted together:
	singleSimData = sim.allSimData
	LFPPops = list(singleSimData['LFPPops'].keys())
	for pop in LFPPops:
		allLFPData['LFPPops'][pop] = singleSimData['LFPPops'][pop]




########################
####### PLOTTING #######
########################

### LFP, CSD, TRACES ### ## CHANGE THESE TO ARGUMENTS ## 
LFP = 0
LFPcellContrib = 0
LFPPopContrib = 1
CSD = 0
traces = 0
waveletNum = 1
#electrodes = [3,5, 7, 9, 11, 13, 15, 17, 19] #[3, 4, 5] #['all']	#[4,5,6,7]  	# CHANGE THIS TO DESIRED ELECTRODES 
electrodes = [4, 6, 8, 10, 12, 14, 16, 18]
waveletImg = 0


#### Plot LFP Pops contribution: 

if LFPPopContrib:
	plt.figure(figsize=(6,8))
	#allData = sim.allSimData

	#LFPPops = list(allData['LFPPops'].keys()) # list of pops recorded from 
	LFPPops = list(allLFPData['LFPPops'].keys())
	LFPPops.remove('NGF1')

	for pop in LFPPops:

		popColorNum = LFPPops.index(pop)
		color = colorList[popColorNum%len(colorList)]
		print('pop: ' + str(pop) + ' color: ' + str(color))

		#lfp = np.array(sim.allSimData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]
		lfp = np.array(allLFPData['LFPPops'][pop])[int(timeRange[0]/sim.cfg.recordStep):int(timeRange[1]/sim.cfg.recordStep),:]


		data = {'lfp': lfp}  # returned data
		ydisp = 0.02 #0.0025 #np.absolute(lfp).max() * 1.0 ## (1.0 --> separation)
		offset = 1.0*ydisp


		first = True ## for labeling purposes!! 
		for i,elec in enumerate(electrodes):
			if isinstance(elec, Number): 
				lfpPlot = lfp[:, elec]  #color = colors[i%len(colors)]
				lw = 1.0

			if first:
				first = False
				plt.plot(t, -lfpPlot+(i*ydisp),  linewidth=lw, label=pop, color = color) #-lfpPlot+(i*ydisp) #color=color,
			else:
				plt.plot(t, -lfpPlot+(i*ydisp),  linewidth=lw, color = color)

			if len(electrodes) > 1:
				plt.text(timeRange[0]-0.07*(timeRange[1]-timeRange[0]), (i*ydisp), elec, ha='center', va='top', fontweight='bold') # fontsize=fontSize, color=color,

		ax = plt.gca()

		data['lfpPlot'] = lfpPlot
		data['ydisp'] =  ydisp
		data['t'] = t

	if len(electrodes) > 1:
		plt.text(timeRange[0]-0.14*(timeRange[1]-timeRange[0]), (len(electrodes)*ydisp)/2.0, 'LFP electrode', color='k', ha='left', va='bottom', rotation=90) # fontSize=fontSize, 
		plt.ylim(-offset, (len(electrodes))*ydisp)
	else:
		plt.suptitle('LFP Signal', fontweight='bold') #fontSize=fontSize, 

	ax.invert_yaxis()
	plt.xlabel('time (ms)') # fontsize=fontSize
	plt.legend()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.get_yaxis().set_visible(False)

	filename = based + 'try_this.png'
	plt.savefig(filename)



## Doesn't matter which file was last to load for sim in this case --> should all be the same except for subsets of LFP cell contrib saved 
# allData = sim.allSimData
# LFPPops = allData['LFPPops'].keys()
if LFP:
	#for pop in LFPPops:
	sim.analysis.plotLFP(plots=['timeSeries'],pop=pop,filtFreq = [13,30],normSignal=True,electrodes=electrodes,showFig=True, timeRange=timeRange), #figSize=(5,5)) # electrodes=[2,6,11,13] # saveFig=figname, saveFig=True, plots=['PSD', 'spectrogram']
if CSD:
	sim.analysis.plotCSD(spacing_um=100, timeRange=timeRange, overlay='LFP', hlines=0, layerLines=1, layerBounds = layerBounds,saveFig=0, figSize=(5,5), showFig=1) # LFP_overlay=True
if traces:
	sim.analysis.plotTraces(include=[(pop, 0) for pop in allpops], timeRange = timeRange, oneFigPer='trace', overlay=True, saveFig=False, showFig=True)#, figSize=(6,8))#figSize=(12,8))




# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

#if __name__ == '__main__':






