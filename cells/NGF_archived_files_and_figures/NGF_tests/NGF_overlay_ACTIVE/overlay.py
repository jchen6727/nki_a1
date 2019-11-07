import test_ngf
import overlay_batch
import numpy as np
import json
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
import os
import numpy

## DATA FROM MACAQUE
data_firing_rates = [21, 24, 27, 29, 33, 35, 36, 38, 40]
amt_above_thresh = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12] #units: nA 


## Find the rheobase of model being tested:
## PARAMS TO MODIFY HERE? ADD AS ARG TO RUNCELL()?? 
test_steps = numpy.arange(0,2,0.01) #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
test_steps = list(test_steps)
global rheobase 
for step in test_steps:
	rheobase = test_ngf.runCell(step)
	plt.close()
	if rheobase >= 0:
		break
print('RHEOBASE:' + str(rheobase))

## Generate current steps to be used for model:
current_steps_to_use = []
for i in range(len(amt_above_thresh)):
	current_steps_to_use.append(rheobase + amt_above_thresh[i])


## GENERATE DATA FROM MODEL
### Delete current data directory
if os.path.isdir('data'):
	shutil.rmtree('data')
	print('REMOVING PRIOR DATA DIRECTORY')

### Run batch 
print('RUNNING BATCH')
overlay_batch.currentSteps(current_steps_to_use)

### Extract data from batch run
model_firing_rates = []
for i in range(len(current_steps_to_use)):
	data = json.load(open('/Users/ericagriffith/Desktop/NGF_tests/NGF_overlay_ACTIVE/data/currentStep__' + str(i) + '.json'))
	num_spikes = data['simData']['spkt']
	freq = len(num_spikes) #sim is for 1000 ms
	model_firing_rates.append(freq)

### Calculate firing rate fitness
firing_rate_fitness = 0
for i in range(len(model_firing_rates)):
	firing_rate_fitness += (model_firing_rates[i] - data_firing_rates[i])**2
print(firing_rate_fitness)

## NOW PLOT DATA 
#plt.close() 
#plt.close()
plt.xlabel('Current above threshold, nA')
plt.ylabel('Steady State Frequency, Hz')
#plt.axis([-0.15,0,-40,5])
plt.plot(amt_above_thresh,data_firing_rates,'bs',label='Blue Square: Empirical')
plt.plot(amt_above_thresh,model_firing_rates,'g^',label='Green Triangle: Model\nFitness: ' + str(round(firing_rate_fitness)))
plt.legend(loc='middle right')
plt.show()