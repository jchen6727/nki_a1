from simdat import *
import os

basedir = '/g100_scratch/userexternal/egriffit/A1/v34_batch_eegSpeech_CINECA_trial_3/' # CINECA

allFiles = os.listdir(basedir)

pklFiles = []
for file in allFiles:
	if '_data.pkl' in file:
		pklFiles.append(file)

for fn in pklFiles:
	fullFilename = basedir + fn

	print('Working with file: ' + fn)

	simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)

	print('loaddat run on ' + fn)

	rasterFilename = fullFilename.split('_data.pkl')[0] + '_raster.png'
	drawraster(dspkT,dspkID,dnumc,tlim=None,msz=2,skipstim=False,drawlegend=False,saveFig=True,rasterFile=rasterFilename) # skipstim=True
	## ^^^ HOW TO SAVE?? 

	print('raster for file ' + fn + ' has been drawn!')