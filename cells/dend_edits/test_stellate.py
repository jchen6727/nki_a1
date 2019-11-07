from netpyne import specs, sim

netParams = specs.NetParams() 
netParams.popParams['TestPop'] = {'cellType': 'ITS', 'numCells': 1, 'cellModel': 'HH_reduced'} #cellType

netParams.importCellParams(label='ITS4_reduced', conds={'cellType':'ITS', 'cellModel':'HH_reduced'}, fileName = 'ITS4_ORIG.py', cellName = 'ITS4_cell')


### V_INIT SECTION ###
#netParams.cellParams['ITS4_reduced']['secs']['soma']['vinit'] = -70.4
#netParams.cellParams['ITS4_reduced']['secs']['dend']['vinit'] = -70.4

## PASSIVE ##
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['pas']['g'] = 3.79e-03 # ORIG: 3.3333333333333335e-05
#netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['pas']['e'] =  # ORIG: -70.0
#netParams.cellParams['ITS4_reduced']['secs']['soma']['geom']['cm'] = # ORIG: 0.75


### ACTIVE ###
## SOMA 
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['naz']['gmax'] = 72000 # ORIG: 30000 
netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['kv']['gbar'] = 1700 # ORIG: 1500 

## DENDRITE
#netParams.cellParams['ITS4_reduced']['secs']['dend']['mechs']['naz']['gmax'] = 1 # ORIG: 15
#netParams.cellParams['ITS4_reduced']['secs']['soma']['mechs']['km']['gbar'] = 5 # ORIG: 0.1 


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': 0.09}  # 0.054
netParams.stimTargetParams['Input->TestPop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'TestPop'}}




cfg = specs.SimConfig()					# object of class SimConfig to store simulation configuration
cfg.duration = 1.5*1e3 #1*1e3 						# Duration of the simulation, in ms
cfg.dt = 0.02								# Internal integration timestep to use
cfg.verbose = 1							# Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.02 			
cfg.filename = 'model_output'  			# Set file output name
cfg.saveJson = True
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True} # Plot recorded traces for this list of cells
cfg.hParams['celsius'] = 37
#cfg.hParams['v_init'] = -85.7 #-85.3

## Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)