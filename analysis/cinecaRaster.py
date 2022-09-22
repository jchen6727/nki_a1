from netpyne import sim 
from matplotlib import pyplot as plt 
import os 


basedir = '../data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_1/'
# fn = 'v34_batch_eegSpeech_CINECA_trial_0_2_data.pkl'
# fn = '../data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_0/v34_batch_eegSpeech_CINECA_trial_0_3_data.pkl'


allFiles = os.listdir(basedir)

pklFiles = []
for file in allFiles:
	if '_data.pkl' in file:
		pklFiles.append(file)


for fn in pklFiles:
	fullFilename = basedir + fn

	sim.load(fullFilename, instantiate=False)
	print('Loaded ' + fn)

	orderBy = ['pop']

	timeRange = [500,4000]

	fig1 = sim.analysis.plotRaster(include=['allCells'], timeRange=timeRange, labels='legend', 
		popRates=False, orderInverse=True, lw=0, markerSize=12, marker='.',  
		showFig=0, saveFig=0, figSize=(9*0.95, 13*0.9), orderBy=orderBy)

	print('RASTER PLOTTED')

	ax = plt.gca()

	[i.set_linewidth(0.5) for i in ax.spines.values()] # make border thinner

	plt.xticks(timeRange, [timeRange[0], timeRange[1]]) #['0', '1'])
	plt.yticks([0, 5000, 10000], [0, 5000, 10000])

	plt.ylabel('Neuron ID') #Neurons (ordered by NCD within each pop)')
	plt.xlabel('Time (s)')

	plt.title('')

	rasterFilename = fn.split('_data.pkl')[0]
	rasterFile = basedir + rasterFilename + '_RASTER.png' #'speechRaster_0_2.png'
	plt.savefig(rasterFile, dpi=300)

	print('RASTER SAVED')