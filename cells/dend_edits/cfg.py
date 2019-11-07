from netpyne import specs

# Simulation options
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
cfg.hParams['v_init'] = -80 #-85.3

# Variable parameters (used in netParams)
cfg.amp = -0.1
