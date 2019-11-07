from netpyne import specs
from netpyne.batch import Batch 

def currentSteps():
	# Create variable of type ordered dictionary (NetPyNE's customized version) 
	params = specs.ODict()

	# fill in with parameters to explore and range of values (key has to coincide with a variable in simConfig) 
	params['amp'] = [0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1, -0.11, -0.12]

	# create Batch object with paramaters to modify, and specifying files to use
	b = Batch(params=params, cfgFile='cfg.py', netParamsFile='netParams.py')
	
	# Set output folder, grid method (all param combinations), and run configuration
	b.batchLabel = 'currentStep_'
	b.saveFolder = 'data'
	b.method = 'grid'
	b.runCfg = {'type': 'mpi_bulletin', 
				'script': 'init.py', 
				'skip': False}

	# Run batch simulations
	b.run()

# Main code
if __name__ == '__main__':
	currentSteps()