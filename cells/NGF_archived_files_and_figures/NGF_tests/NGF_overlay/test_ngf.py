from netpyne import specs, sim


netParams = specs.NetParams() 
netParams.popParams['NGF_pop'] = {'cellType': 'ngf', 'numCells': 1, 'cellModel': 'NGF'}


cellRule = netParams.importCellParams(label='NGF_Rule', conds={'cellType': 'ngf', 'cellModel': 'NGF'}, fileName='ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

## SOMA: 
cellRule['secs']['soma']['vinit'] = -66.9 #-65.6 #-67.2
cellRule['secs']['soma']['mechs']['pas']['g'] = 5e-7 #5e-7 #7e-7
cellRule['secs']['soma']['mechs']['pas']['e'] = -85#-85
cellRule['secs']['soma']['mechs']['hd']['gbar'] = 1e-05
cellRule['secs']['soma']['geom']['cm'] = 1.5 #0.5

## DEND:
cellRule['secs']['dend']['vinit'] = -66.9 #-65.6 #-67.2
cellRule['secs']['dend']['mechs']['pas']['g'] = 3e-4 #2e-4 #7e-7
#cellRule['secs']['dend']['mechs']['pas']['e'] = -85
#cellRule['secs']['dend']['mechs']['hd']['gbar'] = 1e-05 #1e-05
#cellRule['secs']['dend']['geom']['cm'] = 0.5

## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': -0.03} 
netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}

## Simulation options

simConfig = specs.SimConfig()					# object of class SimConfig to store simulation configuration
simConfig.duration = 1.5*1e3 #1*1e3 						# Duration of the simulation, in ms
simConfig.dt = 0.02								# Internal integration timestep to use
simConfig.verbose = 0							# Show detailed messages 
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.02 			
simConfig.filename = 'model_output'  			# Set file output name
simConfig.analysis['plotTraces'] = {'include': [0]} # Plot recorded traces for this list of cells
simConfig.hParams['celsius'] = 37

## Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)
