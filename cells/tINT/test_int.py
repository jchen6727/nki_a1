from netpyne import specs, sim
from neuron import h
h.load_file("stdrun.hoc")

netParams = specs.NetParams() 


## CREATE POP 
netParams.popParams['sTI'] = {'cellType': 'TI', 'numCells': 1, 'cellModel': 'HH_reduced'} 

## Import .hoc file
netParams.importCellParams(label='TI_reduced', conds={'cellType': 'TI', 'cellModel': 'HH_reduced'}, fileName='sTI.hoc', cellName='sTI', importSynMechs=True)
#netParams.importCellParams(label='TI_reduced', conds={'cellType': 'TI', 'cellModel': 'HH_reduced'}, fileName='sTI_large.hoc', cellName='sTI_large', importSynMechs=True)



## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 200, 'dur': 500, 'amp': 0}  # 0.4
netParams.stimTargetParams['Input->sTC'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'sTI'}}


## cfg  
cfg = specs.SimConfig()					# object of class SimConfig to store simulation configuration
cfg.duration = 1*1e3 						# Duration of the simulation, in ms
cfg.dt = 0.25 #0.01 --> 0.02								# Internal integration timestep to use
cfg.verbose = 1							# Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.02 			
cfg.filename = 'model_output'  			# Set file output name
cfg.saveJson = True
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True} # Plot recorded traces for this list of cells
cfg.hParams['celsius'] = 36 # from batch_.hoc, was 37 


# ## Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)