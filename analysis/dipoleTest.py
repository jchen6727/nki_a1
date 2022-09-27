from simdat import *
import os

basedir = '/g100_scratch/userexternal/egriffit/A1/v36_batch_eegSpeech_CINECA_trial_12/' # CINECA
# basedir = '/Users/ericagriffith/Desktop/NEUROSIM/A1/data/simDataFiles/speech/v34_batch_eegSpeech_CINECA_trial_1/'

allFiles = os.listdir(basedir)

pklFiles = []
for file in allFiles:
	if '_data.pkl' in file:
		pklFiles.append(file)


##### FOR INDIVIDUAL TESTING #####
pklFiles = ['v36_batch_eegSpeech_CINECA_trial_12_0_data.pkl']#, 'v36_batch_eegSpeech_CINECA_trial_12_1_data.pkl']



#######################
#### DIPOLE TEST ######
#######################
dipoleMat = 1
if dipoleMat:

	for fn in pklFiles:
		## LOAD DATA 
		fullFilename = basedir + fn
		print('Working with file: ' + fn)
		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)
		print('loaddat run on ' + fn)

		
		# outfn = fullFilename.split('_data.pkl')[0] + '_dipoleMat.mat'
		# save_dipoles_matlab(outfn, simConfig, sdat, dnumc, dstartidx, dendidx)


		#####################
		## GO LINE BY LINE ##
		from scipy import io

		## determine lidx and then break lidx into two parts to make this smaller ## 
		lidx = list(sdat['dipoleCells'].keys())

		lidxLen = len(lidx)
		partialInd = int(lidxLen / 2)

		print('lidx_part1: lidx[0:' + str(partialInd) + ']')
		lidx_part1 = lidx[:partialInd]
		print('lidx_part2: lidx[' + str(partialInd) + ':]')
		lidx_part2 = lidx[partialInd:]


		## determine lty and also break this into two parts to make it smaller ## 
		lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]

		lty_part1 = lty[:partialInd]
		lty_part2 = lty[partialInd:]

		## determine cellPos and also break it into two parts ## 
		cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]

		cellPos_part1 = cellPos[:partialInd]
		cellPos_part2 = cellPos[partialInd:]


		## determine cellDipoles and also break it into two parts ## 
		cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]

		cellDipoles_part1 = cellDipoles[:partialInd]
		cellDipoles_part2 = cellDipoles[partialInd:]


		## create dict w/ matlab data to save ## 
		matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum']}
		
		#### NOTE: not sure what to do about dipoleSum... hmm....
		matDat_part1 = {'cellPos': cellPos_part1, 'cellPops': lty_part1, 'cellDipoles': cellDipoles_part1, 'dipoleSum': sdat['dipoleSum']}
		matDat_part2 = {'cellPos': cellPos_part2, 'cellPops': lty_part2, 'cellDipoles': cellDipoles_part2, 'dipoleSum': sdat['dipoleSum']}


		## SAVE ## 
		outfn_part1 = fullFilename.split('_data.pkl')[0] + '_PART_1_' + '_dipoleMat.mat'
		outfn_part2 = fullFilename.split('_data.pkl')[0] + '_PART_2_' + '_dipoleMat.mat'
		io.savemat(outfn_part1, matDat_part1, do_compression=True)
		print('part 1 saved')
		io.savemat(outfn_part2, matDat_part2, do_compression=True)
		print('part 2 saved')


		############################################################################
		#### ORIGINAL ##### 
		# from scipy import io

		# lidx = list(sdat['dipoleCells'].keys())
		# lty = [GetCellType(idx,dnumc,dstartidx,dendidx) for idx in lidx]

		# cellPos = [GetCellCoords(simConfig,idx) for idx in lidx]
		# cellDipoles = [sdat['dipoleCells'][idx] for idx in lidx]

		# matDat = {'cellPos': cellPos, 'cellPops': lty, 'cellDipoles': cellDipoles, 'dipoleSum': sdat['dipoleSum']}


		### print('dipoles saved to matlab file!')


###########################
#### RASTER PLOTTING ######
###########################
raster = 0

if raster:

	for fn in pklFiles:
		fullFilename = basedir + fn

		print('Working with file: ' + fn)

		simConfig, sdat, dstartidx, dendidx, dnumc, dspkID, dspkT = loaddat(fullFilename)

		print('loaddat run on ' + fn)

		rasterFilename = fullFilename.split('_data.pkl')[0] + '_raster.png'
		drawraster(dspkT,dspkID,dnumc,tlim=None,msz=0.5,skipstim=False,drawlegend=True,saveFig=True,rasterFile=rasterFilename) # skipstim=True
		## ^^^ HOW TO SAVE?? 

		print('raster for file ' + fn + ' has been drawn!')




