from netpyne import specs, sim
#from neuron import h,gui

#h.load_file('stdrun.hoc')
#h.load_file('nrngui.hoc')
#h.load_file('noload.hoc')
#h.load_file('stdgui.hoc')


netParams = specs.NetParams() 

netParams.importCellParams(label='NGF_Rule', conds={'cellType': 'NGF', 'cellModel': 'HH_ngf'}, fileName='cells/ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

netParams.popParams['NGF_pop'] = {'cellType': 'NGF', 'numCells': 1, 'cellModel': 'HH_ngf'}


## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 50, 'dur': 500, 'amp': 10} 
netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}

# Simulation options
simConfig = specs.SimConfig()					# object of class SimConfig to store simulation configuration
simConfig.duration = 1*1e3 			# Duration of the simulation, in ms
simConfig.dt = 0.02				# Internal integration timestep to use
simConfig.verbose = 0			# Show detailed messages 
simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
simConfig.recordStep = 0.02 			# changing BACK to 0.1 from 0.2 to match step size in demo_PY_IB.hoc (1 -- changing to match .dt) Step size in ms to save data (eg. V traces, LFP, etc)
simConfig.filename = 'model_output'  # Set file output name
simConfig.analysis['plotTraces'] = {'include': [0]} # Plot recorded traces for this list of cells


# Create network and run simulation
sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)  
