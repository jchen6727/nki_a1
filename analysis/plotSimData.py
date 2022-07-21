from simDataAnalysis import *


##########################
#### USEFUL VARIABLES ####
##########################
## set layer bounds:
layerBounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}

## Cell populations: 
allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']#, 'IC']
L1pops = ['NGF1']
L2pops = ['IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2']
L3pops = ['IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3']
L4pops = ['ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4']
L5Apops = ['IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A']
L5Bpops = ['IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B']
L6pops = ['IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']
thalPops = ['TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'TIM']
ECortPops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6', 'CT6']
ICortPops = ['NGF1', 
			'PV2', 'SOM2', 'VIP2', 'NGF2', 
			'PV3', 'SOM3', 'VIP3', 'NGF3',
			'PV4', 'SOM4', 'VIP4', 'NGF4',
			'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',
			'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',
			'PV6', 'SOM6', 'VIP6', 'NGF6']
AllCortPops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']
EThalPops = ['TC', 'TCM', 'HTC']				# TEpops = ['TC', 'TCM', 'HTC']
IThalPops = ['IRE', 'IREM', 'TI', 'TIM']		# TIpops = ['IRE', 'IREM', 'TI', 'TIM']
reticPops = ['IRE', 'IREM']
matrixPops = ['TCM', 'TIM', 'IREM']
corePops = ['TC', 'HTC', 'TI', 'IRE']


## electrodes <--> layer
L1electrodes = [0]
L2electrodes = [1]
L3electrodes = [2,3,4,5,6,7,8,9]
L4electrodes = [9,10,11,12]
L5Aelectrodes = [12,13]
L5Belectrodes = [13,14,15]
L6electrodes = [15,16,18,19]
allElectrodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
supra = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gran = [9, 10, 11, 12]
infra = [12, 13, 14, 15, 16, 17, 18, 19]

## for COLORS in DATA PLOTTING!! 
colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
			[0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
			[0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
			[0.71,0.82,0.41], [0.0,0.2,0.5], [0.70,0.32,0.10]]*3

colorDict = {}
for p in range(len(allpops)):
	colorDict[allpops[p]] = colorList[p]



# colorDictCustom = {}
# colorDictCustom['ITP4'] = 'magenta'	# 'tab:purple'
# colorDictCustom['ITS4'] = 'dodgerblue' 	# 'tab:green'
# colorDictCustom['IT5A'] = 'lime'	# 'tab:blue'





###################################
###### COMBINED LFP PLOTTING ######
###################################


### NEED TO DEFINE:
#### timeRange
#### colorDict -- DONE 

plotLFPCombinedData = 0


if plotLFPCombinedData:

	includePops = ['IT3'] #, 'IT5A', 'PT5B']  			## Automate this somehow?? 
	electrode = [10]									## Automate this somehow?? 


	for pop in includePops:

		print('Plotting LFP spectrogram and timeSeries for ' + pop + ' at electrode ' + str(electrode))

		## Get dictionaries with LFP data for spectrogram and timeSeries plotting  
		LFPSpectOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['spectrogram'], electrode=electrode) 
		LFPtimeSeriesOutput = getLFPDataDict(dataFile, pop=pop, timeRange=timeRange, plotType=['timeSeries'], electrode=electrode)

		plotCombinedLFP(timeRange=timeRange, pop=pop, colorDict=colorDict, plotTypes=['timeSeries'], 
			spectDict=LFPSpectOutput, timeSeriesDict=LFPtimeSeriesOutput, figSize=(10,7), colorMap='jet', 
			minFreq=15, maxFreq=None, vmaxContrast=None, electrode=electrode, savePath=None, saveFig=True)



#####################
######## CSD ########
#####################




##########################################
###### COMBINED SPIKE DATA PLOTTING ######
##########################################




################################
###### peakF calculations ######
################################