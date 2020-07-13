import utils

allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4',
'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC']

timeRange=[450,700]
sim = utils.loadFromFile('../data/v25_batch4/v25_batch4_1_1.json')

#sim.analysis.getCSD(minf=0.1, maxf=100, spacing_um=100, sampr=1./sim.cfg.recordStep)

#sim.analysis.plotLFP(electrodes=[0,1,2,3], timeRange=timeRange, plots=['timeSeries'], saveFig='../data/v23_batch8/v23_batch8_0_0_0_0_%d_LFP.png' % (i))
#sim.analysis.plotLFP(electrodes=[0,1,2,3], timeRange=timeRange, plots=['spectrogram'], saveFig='../data/v23_batch8/v23_batch8_0_0_0_0_%d_LFP_spect.png' % (i))

# sim.analysis.plotCSD(electrodes=range(11), timeRange=timeRange, separation=1.2, plots=['timeSeries'], smooth=512, saveFig=' % (i))

sim.analysis.plotCSD(empirical=True,  NHP =False, timeRange=timeRange, spacing_um=100, hlines=True, saveData=None, saveFig='../data/v25_batch4/v25_batch4_1_1_CSD.png', showFig=True, LFP_overlay=True)
