from netpyne import specs, sim
import matplotlib.pyplot as plt


def runCell(currentStep):
	netParams = specs.NetParams() 
	netParams.popParams['NGF_pop'] = {'cellType': 'ngf', 'numCells': 1, 'cellModel': 'NGF'}

	cellRule = netParams.importCellParams(label='NGF_Rule', conds={'cellType': 'ngf', 'cellModel': 'NGF'}, fileName='ngf_cell.hoc', cellName='ngfcell', importSynMechs = True)

	## PASSIVE SOMA PARAMS
	#cellRule['secs']['soma']['vinit'] = -66.9 #-65.6
	# cellRule['secs']['soma']['mechs']['pas']['g'] =  5e-7 #5e-7
	# cellRule['secs']['soma']['mechs']['pas']['e'] = -85
	# cellRule['secs']['soma']['mechs']['hd']['gbar'] = 1e-05
	# cellRule['secs']['soma']['geom']['cm'] = 1.5 #0.5
	# # #print(cellRule['secs']['soma']['mechs']['ch_KvAngf']['gmax'])

	# # ## PASSIVE DEND PARAMS
	# # #cellRule['secs']['dend']['vinit'] = -66.9 #-65.6
	# cellRule['secs']['dend']['mechs']['pas']['g'] = 3e-4
	
	# # ### SOMA ACTIVE -- MODIFY THESE PARAMS HERE *AND IN NETPARAMS*
	# cellRule['secs']['soma']['mechs']['ch_Navngf']['gmax'] = 0.1 #3.7860265
	# cellRule['secs']['soma']['mechs']['ch_Kdrfastngf']['gmax'] = 0.09 #0.15514516

	# # ### DEND ACTIVE -- MODIFY THESE PARAMS HERE *AND IN NETPARAMS*
	# cellRule['secs']['dend']['mechs']['ch_Navngf']['gmax'] = 3.7860265
	# cellRule['secs']['dend']['mechs']['ch_Kdrfastngf']['gmax'] = 0.03 #0.15514516
	
	## Add in Stimulation Source (IClamp) 
	netParams.stimSourceParams['Input'] = {'type': 'IClamp', 'del': 250, 'dur': 1000, 'amp': currentStep}
	netParams.stimTargetParams['Input->NGF_pop'] = {'source': 'Input', 'sec':'soma', 'loc': 0.5, 'conds': {'pop':'NGF_pop'}}

	## Simulation options
	simConfig = specs.SimConfig()					# object of class SimConfig to store simulation configuration
	simConfig.duration = 1.5*1e3 						# Duration of the simulation, in ms
	simConfig.dt = 0.02								# Internal integration timestep to use
	simConfig.verbose = 0							# Show detailed messages 
	simConfig.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
	simConfig.recordStep = 0.02 			
	simConfig.filename = 'model_output'  			# Set file output name
	simConfig.analysis['plotTraces'] = {'include': [0]} # Plot recorded traces for this list of cells
	simConfig.hParams['celsius'] = 37
	#simConfig.hParams['v_init'] = -65.6

	## Create network and run simulation
	sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

	data = sim.allSimData
	spkt = data['spkt']
	if len(spkt) >= 1:
		return currentStep
	else: 
		print('Not at threshold')
		return -1000

	#plt.close()


# Main code
if __name__ == '__main__':
	runCell(currentStep)




