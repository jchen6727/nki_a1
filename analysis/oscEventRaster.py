from netpyne import sim 
from matplotlib import pyplot as plt 
import os 
import numpy as np


basedir = '../data/oscEventRasters/pklFiles/'
# fn = '../data/oscEventRasters/pklFiles/A1_v34_batch65_v34_batch65_0_0_data.pkl'


# allFiles = os.listdir(basedir)

# pklFiles = []
# for file in allFiles:
# 	if '_data.pkl' in file:
# 		pklFiles.append(file)


waveletInfo = {
	'delta': {'dataFile': 'A1_v34_batch65_v34_batch65_0_0_data.pkl', 'timeRange': [1480, 2520], 'dataFile2': 'v34_batch57_3_4_data.pkl'},	#'channel': 14}, # [1484.9123743729062, 2512.2209353489225]
	'beta': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl',	'timeRange': [456, 572], 'dataFile2': 'v34_batch57_3_2_data.pkl'},		#'channel': 14}, # [455.80379839663993, 571.4047617460291]
	'alpha': {'dataFile': 'A1_v34_batch67_v34_batch67_0_0_data.pkl', 'timeRange': [3111, 3325], 'dataFile2': 'v34_batch57_3_2_data.pkl'}, 	# NOTE: don't forget 6_11 timeRange	#'channel': 9}, # [3111.0311106222125, 3324.733247664953]
	'theta': {'dataFile': 'A1_v34_batch67_v34_batch67_1_1_data.pkl', 'timeRange': [2785, 3350], 'dataFile2': 'v34_batch57_3_3_data.pkl'}, 	#'channel': 8}} # [2785.22321038684, 3347.9278996316607]
	'gamma': {'dataFile': 'v34_batch57_4_4_data.pkl', 'timeRange': [3890, 3960], 'dataFile2': 'v34_batch57_4_4_data.pkl'}}	# gamma timeRange: [3895.632463874398, 3957.13297638294]


freqBands = list(waveletInfo.keys())

batch57 = 1
saveFigs = 0
markerType = 'o'

if batch57: 
	for band in freqBands:
		fn = basedir + waveletInfo[band]['dataFile2']
		sim.load(fn, instantiate=False)
		print('Loaded ' + fn)

		orderBy = ['pop']


		timeRange = waveletInfo[band]['timeRange']


		sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels=False,
			popRates=False, orderInverse=True, lw=0, markerSize=12, marker=markerType,  
			showFig=0, saveFig=0, orderBy=orderBy)		# labels='legend' # figSize=(9*0.95, 13*0.9)


		plt.ylabel('Neuron ID') # Neurons (ordered by NCD within each pop)')
		plt.xlabel('Time (ms)')

		plt.title(band + ' osc event raster')

		print(band + ' RASTER PLOTTED')


		## SAVING ##
		if saveFigs:
			rasterFilename = band + '_dataFile2_' + waveletInfo[band]['dataFile2'] 
			rasterFile = basedir + rasterFilename + '_RASTER.png'
			plt.savefig(rasterFile, dpi=300)
			print('RASTER SAVED')
		else:
			plt.show()


else:
	for band in freqBands:
		fn = basedir + waveletInfo[band]['dataFile']
		sim.load(fn, instantiate=False)
		print('Loaded ' + fn)

		orderBy = ['pop']


		timeRange = waveletInfo[band]['timeRange'] 			#[0,sim.cfg.duration]#[500,sim.cfg.duration]#[500,10000]

		sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels='legend', 
			popRates=False, orderInverse=True, lw=0, markerSize=12, marker='.',  
			showFig=0, saveFig=0, orderBy=orderBy)		# figSize=(9*0.95, 13*0.9)

		plt.ylabel('Neuron ID') # Neurons (ordered by NCD within each pop)'
		plt.xlabel('Time (ms)')

		plt.title('dataFile !')


		print(band + ' RASTER PLOTTED')


		## SAVING ## 
		if saveFigs:
			rasterFilename = band + '_dataFile_' + waveletInfo[band]['dataFile']
			rasterFile = basedir + rasterFilename + '_RASTER.png'
			plt.savefig(rasterFile, dpi=300)
			print('RASTER SAVED')
		else:
			plt.show()


