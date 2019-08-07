from netpyne import specs 

#Simulation optios 
cfg = specs.SimConfig() # usually do simConfig (but will do cfg here bc need to for batch run)

cfg.duration = 1.1*1e3 			# Duration of the simulation, in ms
cfg.dt = 0.05 				# Internal integration timestep to use # lowering this from 0.1 to match record Step? 
cfg.verbose = 0			# Show detailed messages 
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.05	# 0.1 orig # 0.02 to match abf file for Martin's protocol # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.filename = 'model_output'  # Set file output name
cfg.savePickle = False 		# Save params, network and sim output to pickle file
cfg.saveJson = True
#cfg.hParams['celsius'] = 24 

cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True} 

# Variable Parameters (used in netParams)
cfg.amp = 0.04 # instantiates this as a variable that can be changed in batch file 