## IMPORTS ##
import cell_batch
import os
import shutil
import json
import matplotlib.pyplot as plt

current_steps = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11] 

### Delete data directory 
# if os.path.isdir('data_old'):
# 	shutil.rmtree('data_old')
# 	print('REMOVING PRIOR DATA DIRECTORY')

## Run batch of suprathreshold current steps on unmodified cell types 
print('RUNNING BATCH: OLD, ACTIVE')
cell_batch.batch_compare('OLD', 'active')


## Extract Firing Rate
old_firing_rates = []

for i in range(len(current_steps)):
	data = json.load(open('/Users/ericagriffith/Desktop/NEUROSIM/A1/cells/dend_edits/data_old/compare__' + str(i) + '.json')) # <<---- ?????
	num_spikes = data['simData']['spkt']
	freq = len(num_spikes) #sim is for 1000 ms
	old_firing_rates.append(freq)


## NEW
### Delete current data directory
if os.path.isdir('data_new'):
	shutil.rmtree('data_new')
	print('REMOVING PRIOR DATA DIRECTORY')

## Run batch of suprathreshold current steps on modified cell types 
print('RUNNING BATCH: NEW, ACTIVE')
cell_batch.batch_compare('NEW', 'active')


## Extract Firing Rate 
new_firing_rates = []

for i in range(len(current_steps)):
	data = json.load(open('/Users/ericagriffith/Desktop/NEUROSIM/A1/cells/dend_edits/data_new/compare__' + str(i) + '.json')) # <<---- ?????
	num_spikes = data['simData']['spkt']
	freq = len(num_spikes) #sim is for 1000 ms
	new_firing_rates.append(freq)



## COMPARISON
print('OLD FIRING RATES:' + str(old_firing_rates))
print('NEW FIRING RATES:' + str(new_firing_rates))

fitness = []
for i in range(len(old_firing_rates)):
	fitness.append((old_firing_rates[i] - new_firing_rates[i])**2)
fitness_score = 0
for i in range(len(fitness)):
	fitness_score += fitness[i]
print(fitness_score)

plt.xlabel('Current Step, nA')
plt.ylabel('Steady State Frequency, Hz')
plt.title('ITS4 ACTIVE') ## CHANGE THIS 
plt.plot(current_steps,old_firing_rates,'bs',label='Blue Square: Unmodified')
plt.plot(current_steps,new_firing_rates,'g^',label='Green Triangle: Modified\nFITNESS: ' + str(fitness_score))
plt.legend(loc='middle right')
plt.show()

