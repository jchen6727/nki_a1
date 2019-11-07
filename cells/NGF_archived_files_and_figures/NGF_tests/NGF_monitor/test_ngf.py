from netpyne import specs, sim


netParams = specs.NetParams() 
netParams.popParams['NGF_pop'] = {'cellType': 'ngf', 'numCells': 1, 'cellModel': 'NGF'}


netParams.importCellParams(label='NGF_Rule', conds={'cellType': 'ngf', 'cellModel': 'NGF'}, fileName='ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

#netParams.popParams['VIP_NGF_pop'] = {'cellType': 'vipcr', 'numCells': 1, 'cellModel': 'HH_vip_ngf'}


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 50, 'dur': 1000, 'amp': -0.1} 
netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}

## Simulation options

simConfig = specs.SimConfig()					# object of class SimConfig to store simulation configuration
simConfig.duration = 1.1*1e3 #1*1e3 						# Duration of the simulation, in ms
simConfig.dt = 0.02								# Internal integration timestep to use
simConfig.verbose = 0							# Show detailed messages 
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.02 			
simConfig.filename = 'model_output'  			# Set file output name
simConfig.analysis['plotTraces'] = {'include': [0]} # Plot recorded traces for this list of cells


## Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)
