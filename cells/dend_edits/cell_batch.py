from netpyne import specs
from netpyne.batch import Batch 

### MODIFY THIS 
def batch_compare(ComparisonType, FitType):
	# Create variable of type ordered dictionary (NetPyNE's customized version) 
	params = specs.ODict()

	# fill in with parameters to explore and range of values (key has to coincide with a variable in simConfig) 
	if FitType == 'active':
		params['amp'] = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11] 
	elif FitType == 'passive':
		params['amp'] = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]

	# Set output folder, grid method (all param combinations), and run configuration
	if ComparisonType == 'OLD':
		# create Batch object with paramaters to modify, and specifying files to use
		b = Batch(params=params, cfgFile='cfg.py', netParamsFile='netParams_OLD.py')
		b.batchLabel = 'compare_'
		b.saveFolder = 'data_old'
		b.method = 'grid'
		b.runCfg = {'type': 'mpi_bulletin', 
				'script': 'init.py', 
				'skip': True}
		
	elif ComparisonType == 'NEW':
		b = Batch(params=params, cfgFile='cfg.py', netParamsFile='netParams_NEW.py')
		b.batchLabel = 'compare_'
		b.saveFolder = 'data_new'
		b.runCfg = {'type': 'mpi_bulletin', 
				'script': 'init.py', 
				'skip': False}
	
	# Run batch simulations
	b.run()

# Main code
if __name__ == '__main__':
	batch_compare(ComparisonType, FitType)