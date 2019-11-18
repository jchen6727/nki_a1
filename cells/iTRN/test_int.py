from netpyne import specs, sim
from neuron import h
h.load_file("stdrun.hoc")

netParams = specs.NetParams() 


## CREATE POP 
#netParams.popParams['sTC'] = {'cellType': 'TC', 'numCells': 1, 'cellModel': 'HH_reduced'} 

## Import .hoc file
#netParams.importCellParams(label='TC_reduced', conds={'cellType': 'TC', 'cellModel': 'HH_reduced'}, fileName='cells/sTC.py', cellName='sTC', importSynMechs=True)



## Add in Stimulation Source (IClamp) 
#netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': 0}  # 0.4
#netParams.stimTargetParams['Input->sTC'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'sTC'}}


## cfg  
cfg = specs.SimConfig()					# object of class SimConfig to store simulation configuration
cfg.duration = 9*1e3 #1*1e3 						# Duration of the simulation, in ms
cfg.dt = 0.01								# Internal integration timestep to use
cfg.verbose = 1							# Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.02 			
cfg.filename = 'model_output'  			# Set file output name
cfg.saveJson = True
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True} # Plot recorded traces for this list of cells
cfg.hParams['celsius'] = 37


## Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)