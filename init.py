"""
init.py

Starting script to run NetPyNE-based A1 model.


Usage:
    python init.py # Run simulation, optionally plot a raster


MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py


Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

#import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

from netpyne import sim

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')
# sim.initialize(netParams, cfg)
# sim.net.createCells()
# sim.gatherData()
#sim.create(netParams, cfg)
#sim.gatherData()
sim.createSimulateAnalyze(netParams, cfg)

# sim.analysis.plotLFP(plots=['timeSeries'], timeRange =[100,600], saveFig=True, electrodes=['avg', 1,2,3,4,5], figSize=(6,12), showFig=0)

# sim.analysis.plotLFP(plots=['PSD'], timeRange =[100,600], saveFig=True, electrodes=['avg', 1,2,3,4,5], figSize=(12,7), showFig=0, overlay=1, maxFreq=80, normPSD=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig=True, electrodes=['avg', 1,2,3,4,5], figSize=(6,18), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp0.png', electrodes=[0], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp1.png', electrodes=[1], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp2.png', electrodes=[2], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp3.png', electrodes=[3], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp4.png', electrodes=[4], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotLFP(plots=['spectrogram'], timeRange =[100,600], saveFig='lfp5.png', electrodes=[5], figSize=(11,5), showFig=0, maxFreq=80, normSpec=1)

# sim.analysis.plotRaster(**{'include': allpops, 'saveFig': True, 'showFig': False, 'popRates': False, 'orderInverse': True, 'timeRange': [0,cfg.duration], 'figSize': (15,12), 'lw': 0.3, 'markerSize':10, 'marker': '.', 'dpi': 300, 'timeRange':[100,600]})

# sim.analysis.plotTraces(**{'include': [158, 3648, 3740], 'oneFigPer': 'trace', 'overlay': True, 'saveFig': True, 'showFig': False, 'figSize':(12,4)})



, 'PSD', 