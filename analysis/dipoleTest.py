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

	simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(pklFile)

	drawraster(dspkT,dspkID,tlim=None,msz=2,skipstim=False,drawlegend=False) # skipstim=True
	## ^^^ HOW TO SAVE?? 