## IMPORTS ##
import cell_batch
import os
import shutil
import json
from statistics import mean
import matplotlib.pyplot as plt

current_steps = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08]
### OLD
### Delete current data directory
#if os.path.isdir('data_old'):
#	shutil.rmtree('data_old')
#	print('REMOVING PRIOR DATA DIRECTORY')

## Run batch of subthreshold current steps on unmodified cell types 
print('RUNNING BATCH: OLD, PASSIVE')
cell_batch.batch_compare('OLD', 'passive')



old_responses = []
for i in range(len(current_steps)):
	data = json.load(open('/Users/ericagriffith/Desktop/NEUROSIM/A1/cells/dend_edits/data_old/compare__' + str(i) + '.json'))
	vdata_total = data['simData']['V_soma']['cell_0']
	vdata = vdata_total[12500:62500]
	mean_responses = mean(vdata)
	old_responses.append(mean_responses)


### NEW
### Delete current data directory
if os.path.isdir('data_new'):
	shutil.rmtree('data_new')
	print('REMOVING PRIOR DATA DIRECTORY')

## Run batch of subthreshold current steps on modified cell types 
print('RUNNING BATCH: NEW, PASSIVE')
cell_batch.batch_compare('NEW', 'passive')


new_responses = []
for i in range(len(current_steps)):
	data = json.load(open('/Users/ericagriffith/Desktop/NEUROSIM/A1/cells/dend_edits/data_new/compare__' + str(i) + '.json'))
	vdata_total = data['simData']['V_soma']['cell_0']
	vdata = vdata_total[12500:62500]
	mean_responses = mean(vdata)
	new_responses.append(mean_responses)


## COMPARISON
print('OLD STEADY STATES:' + str(old_responses))
print('NEW STEADY STATES:' + str(new_responses))

fitness = []
for i in range(len(old_responses)):
	fitness.append((old_responses[i] - new_responses[i])**2)
fitness_score = 0
for i in range(len(fitness)):
	fitness_score += fitness[i]
print(fitness_score)

plt.xlabel('Current Step, nA')
plt.ylabel('Mean Steady State Response, mV')
plt.title('ITS4 PASSIVE') ## CHANGE THIS 
plt.plot(current_steps,old_responses,'bs',label='Blue Square: Unmodified')
plt.plot(current_steps,new_responses,'g^',label='Green Triangle: Modified\nFITNESS: ' + str(fitness_score))
plt.legend(loc='middle right')
plt.show()