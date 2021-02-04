"""
init.py

Starting script to run NetPyNE-based A1 model.


Usage:
    python init.py # Run simulation, optionally plot a raster


MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py


Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers

from netpyne import sim

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')

# sim.createSimulateAnalyze(netParams, cfg)

sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                      			# run parallel Neuron simulation  
sim.gatherData()                  			# gather spiking data and cell info from each node
sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()         			# plot spike raster etc

layer_bounds= {'L1': 100, 'L2': 160, 'L3': 950, 'L4': 1250, 'L5A': 1334, 'L5B': 1550, 'L6': 2000}
filename = sim.cfg.saveFolder+'/'+sim.cfg.simLabel

sim.analysis.plotRaster(**{'include': sim.cfg.allpops, 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'timeRange': [1500,2500], 'figSize': (14,12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300})
sim.analysis.plotSpikeStats(stats=['rate'],figSize = (6,12), timeRange=[0, 2500], dpi=300, showFig=0, saveFig=1)
    
for elec in [1, 6, 11, 16]:
    sim.analysis.plotLFP(**{'plots': ['timeSeries'], 'electrodes': [elec], 'timeRange': [1500, 2500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_signal_1s-2s_elec_'+str(elec), 'showFig': False})
    sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [1500, 2500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_PSD_1s-2s_elec_'+str(elec), 'showFig': False})
    sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [1500, 2500], 'maxFreq':80, 'figSize': (8,4), 'saveData': False, 'saveFig': filename[:-4]+'_LFP_spec_1s-2s_elec_'+str(elec), 'showFig': False})

tranges = [[2000, 2200],
            [2000, 2100]]
    #         [1980, 2100],
    #         [2080, 2200],
    #         [2030, 2150],
    #         [2100, 2200]] 
for t in tranges:
    sim.analysis.plotCSD(**{'spacing_um': 100, 'overlay': 'LFP', 'layer_lines': 1, 'layer_bounds': layer_bounds, 'timeRange': [t[0], t[1]], 'saveFig': filename[:-4]+'_CSD_%d-%d' % (t[0], t[1]), 'figSize': (6,9), 'dpi': 300, 'showFig': 0})
