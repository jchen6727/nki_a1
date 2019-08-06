###### IMPORTS ######
from random import Random         	# pseudorandom number generation
from inspyred import ec           	# evolutionary algorithm
import numpy
from scipy import interpolate     	# for curve smoothing in adaptation fitness
import scipy.io                   	# for importing matlab files
from netpyne import sim 			# neural network design and simulation
import NGF_batch	        		# Import for BATCH RUN
from statistics import mean 		# to calculate mean(s) for RMP fitness function
import json                       	# to read batch output files
import os  							# to be able to navigate file system in python
import tarfile						# to compress data directory
import shutil						# also necessary to compress data directories
import datetime						# to label data directories for compression
from time import sleep,time							# To pause program while batch runs finish



###### GENERATE PARAMETERS ######
# Design parameter generator function, used in the ec evolve function --> final_pop = my_ec.evolve(generator=generate_netparams,...)
def generate_netparams(random, args):
    size = args.get('num_inputs')
    initialParams = [random.uniform(minParamValues[i], maxParamValues[i]) for i in range(size)]
    return initialParams



###### FITNESS FUNCTION ######
# Design fitness function, used in the ec evolve function --> final_pop = my_ec.evolve(...,evaluator=evaluate_netparams,...)
def evaluate_netparams(candidates, args):
	global ngen   # This will keep track of the number of generations 
	ngen +=1

############################################################################################################

	###### SUBMIT BATCH JOBS FOR ALL CANDIDATES IN THE CURRENT GENERATION ######

############################################################################################################

	print('BATCH SUBMISSIONS FOR GEN ' + str(ngen) + ' ARE BEGINNING')

	for icand,cand in enumerate(candidates): 
		# Modify network params in NGF_netParams.py based on this candidate's params ("genes")

############## CHANGE THESE BASED ON WHAT NEEDS TO BE MODIFIED ###########

		## read the lines in the file
		fo = open('NGF_netParams.py', 'r')
		lines = fo.readlines()
		fo.close()
		## overwrite old lines
		fo = open('NGF_netParams.py', 'w')
		for line in lines: # CHANGE THIS TO MATCH NUMBER OF MODIFIED PARAMS
			if line != lines[len(lines)-1] and line != lines[len(lines)-2] and line != lines[len(lines)-3] and line != lines[len(lines)-4] and line != lines[len(lines)-5]: # requires that there be the desired lines already at the end
				fo.write(line)
		fo.close()

		## append new lines  
		with open('NGF_netParams.py', 'a') as fo: # CHANGE THIS TO REFLECT THE MECHANISMS BEING CHANGED
			fo.write("cellRule['NGF_Rule']['secs']['soma']['mechs'][][]=" + str(cand[0]) + "\ncellRule['NGF_Rule']['secs']['soma']['mechs'][][] =" + str(cand[1]) + "\ncellRule['NGF_Rule']['secs']['soma']['mechs'][][] = " + str(cand[2]) + "\ncellRule['NGF_Rule']['secs']['soma']['mechs'][][] = " + str(cand[3]) + "\ncellRule['NGF_Rule']['secs']['soma']['mechs'][][] = " + str(cand[4]))

		# Run batch using the above candidate params
		NGF_batch.batch_full(icand, ngen, runType)
		print('BATCH JOB FOR CAND_' + str(icand) + ' FROM GEN_' + str(ngen) + ' SUBMITTED')

	print('END OF BATCH SUBMISSIONS FOR GEN ' + str(ngen))


############################################################################################################

	###### CALCULATE FITNESS AS BATCH RUNS COMPLETE ######

############################################################################################################

#### CHANGE FITNESS FUNCTIONS AS NEEDED #### 

	global data_path_stem
	data_path = data_path_stem + str(ngen) + '/'

	cands_completed = 0
	total_cands = len(candidates)
	fitnessCandidates = [None for cand in candidates]


	while cands_completed < total_cands:
		print(str(cands_completed) + ' / ' + str(total_cands) + ' candidates completed')

		unfinished = [i for i, x in enumerate(fitnessCandidates) if x is None]

		for cand_index in unfinished:

			data_file_root = 'NGF_batch_data_cand' + str(cand_index)
			data_files_DONE = os.listdir(data_path) # EXISTING DATA FILES 
			data_files_NEEDED = [data_file_root + '_0.json', data_file_root + '_1.json', data_file_root + '_2.json', data_file_root + '_3.json', data_file_root + '_4.json', data_file_root + '_5.json', data_file_root + '_6.json', data_file_root + '_7.json', data_file_root + '_8.json', data_file_root + '_9.json', data_file_root + '_10.json', data_file_root + '_11.json', data_file_root + '_12.json']
			data_files_DONE.sort()
			data_files_NEEDED.sort()


			if data_files_DONE == data_files_NEEDED:
				print('BATCH RUN FOR CAND ' + str(cand_index) + ' IS COMPLETE')
				print('FITNESS CALCULATIONS FOR CAND ' + str(cand_index) + ' BEGINNING')

				######## FITNESS CALCULATIONS ##########################
				###### RMP ######
				outputData_RMP = json.load(open(data_path + 'NGF_batch_data_cand' + str(cand_index) + '_0.json'))
				vdata_RMP = list(outputData_RMP['simData']['V_soma']['cell_0'])
				RMP_sim = mean(vdata_RMP)
				RMP_fitness = abs(RMP_sim-RMP)^2 # RMP defined in main code



				fitnessCandidates[cand_index] = ec.emo.Pareto([RMP_fitness], maximize = [False, False])

				cands_completed += 1
				print('FITNESS EVALUATIONS FOR CANDIDATE ' + str(cand_index) + ' ARE COMPLETE')

			else:
				sleep(1)
				print('Candidate ' + str(cand_index) + ' is unfinished')


	print('EVALUATION OF GENERATION ' + str(ngen) + ' IS COMPLETE')
	print(fitnessCandidates)
	return fitnessCandidates


# Generation tracking
global ngen
ngen = -1

# SET DATA PATH HERE 
machine_ID = input('COMET or LOCAL or zn?')

if machine_ID == 'COMET':
	data_path_stem = '/oasis/scratch/comet/eyg42/temp_project/A1/NGF_passive/NGF_batch_data_gen_'
	runType = 'hpc_slurm'
elif machine_ID == 'LOCAL':
	data_path_stem = '/Users/ericagriffith/Desktop/NEUROSIM/A1/cells/NGF_optimization/NGF_passive/NGF_batch_data_gen_'
	runType = 'mpi_bulletin'
elif machine_ID == 'zn':
	data_path_stem = '/u/ericag/A1/cells/NGF_optimization/NGF_passive/NGF_batch_data_gen_'
	runType = 'mpi_bulletin'
else:
	raise Exception("Computing system not recognized")


# create random seed for evolutionary computation algorithm --> my_ec = ec.EvolutionaryComputation(rand)
rand = Random()
rand.seed(1)

# OPTIMIZATION TARGETS
subthreshold_deltas = [-6, -9. -13, -16, -19, -23, -27, -29, -30, -35, -36, -38] # changes from RMP in response to subthreshold current step inputs
RMP = -67 #+/- 5mV 

# min and max allowed value for each param optimized:
#
minParamValues = []
maxParamValues = []
## ^^ CHANGE THESE

# instantiate MO evolutionary computation algorithm with random seed
my_ec = ec.emo.NSGA2(rand) #ec.EvolutionaryComputation(rand)

# establish parameters for the evolutionary computation algorithm, additional documentation can be found @ pythonhosted.org/inspyred/reference.html
my_ec.selector = ec.selectors.tournament_selection  # tournament sampling of individuals from population (<num_selected> individuals are chosen based on best fitness performance in tournament)

#toggle variators
my_ec.variator = [ec.variators.uniform_crossover,   # biased coin flip to determine whether 'mom' or 'dad' element is passed to offspring design
                 ec.variators.gaussian_mutation]    # gaussian mutation which makes use of bounder function as specified in --> my_ec.evolve(...,bounder=ec.BOunder(minParamValues, maxParamValues),...)

my_ec.replacer = ec.replacers.generational_replacement    # existing generation is replaced by offspring, with elitism (<num_elites> existing individuals will survive if they have better fitness than offspring)

my_ec.terminator = ec.terminators.evaluation_termination  # termination dictated by number of evaluations that have been run


#call evolution iterator
final_pop = my_ec.evolve(generator=generate_netparams,  # assign design parameter generator to iterator parameter generator
                      evaluator=evaluate_netparams,     # assign fitness function to iterator evaluator
                      pop_size=10,                      # each generation of parameter sets will consist of pop_size individuals
                      maximize=False,                   # best fitness corresponds to minimum value
                      bounder=ec.Bounder(minParamValues, maxParamValues), # boundaries for parameter set ([probability, weight, delay])
                      max_evaluations=50,             	# evolutionary algorithm termination at max_evaluations evaluations
                      num_selected=5,                  	# number of generated parameter sets to be selected for next generation
                      mutation_rate=0.2,                # rate of mutation
                      num_inputs=?????,              		# len([a, b, c, d, ...]) -- number of parameters being varied
                      num_elites=1)                     # 1 existing individual will survive to next generation if it has better fitness than an individual selected by the tournament selection


final_arc = my_ec.archive                               # seen this MO examples
print('Best Solutions: \n')                             # taken from forest_optimization.py example
# print final_arc
print(final_pop)

## Write the best candidates and their corresponding fitness values to the file best_cands.txt
filename = 'best_cands.txt'
file = open(filename,'a') # 'a' will append vs overwrite lines in the text file
for f in final_pop:
  print(f.candidate)
  print(f.fitness)
  file.write('\n' + str(f.candidate))
  file.write('\n' + str(f.fitness))
file.close()
