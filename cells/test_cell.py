from netpyne import specs, sim

netParams = specs.NetParams() 
netParams.popParams['TestPop'] = {'cellType': 'IT', 'numCells': 1, 'cellModel': 'HH_reduced'} #cellType
#netParams.popParams['ITP4'] = {'cellType': 'IT', 'numCells': 1, 'cellModel': 'HH_reduced'} #cellType

#netParams.loadCellParamsRule(label='ITP4_reduced', fileName='ITP4_reduced_renorm_cellParams.pkl') #cellName
netParams.loadCellParamsRule(label='IT5A_reduced', fileName='IT5A_reduced_cellParams.pkl')

### V_INIT SECTION ###
# netParams.cellParams['IT5A_reduced']['globals']['v_init'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['soma']['vinit'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['Adend1']['vinit'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['Adend2']['vinit'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['Adend3']['vinit'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['Bdend']['vinit'] = -85.8
# netParams.cellParams['IT5A_reduced']['secs']['axon']['vinit'] = -85.8

# ## PASSIVE ##
#netParams.cellParams['IT5A_reduced']['secs']['soma']['mechs']['pas']['g'] = # ORIG: 9.442294539558377e-05
#netParams.cellParams['IT5A_reduced']['secs']['soma']['mechs']['pas']['e'] = 

# ## ACTIVE ##
#netParams.cellParams['IT5A_reduced']['secs']['soma']['mechs']['nax']['gbar'] = 	# ORIG: 0.0768253702194
# netParams.cellParams['IT5A_reduced']['secs']['soma']['mechs']['kdr']['gbar'] =  	# ORIG: 0.00833766634808
# netParams.cellParams['IT5A_reduced']['secs']['soma']['mechs']['kdr']['vhalfn'] = # ORIG: 11.6427471384


# ## Add in Stimulation Source (IClamp) 
netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': 0.3} 
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
