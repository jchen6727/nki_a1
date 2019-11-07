import overlay_batch
import numpy as np
import json
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
import os

## (1) DATA FROM MACAQUE
subthreshold_deltas = [-6, -9, -13, -16, -19, -23, -27, -29, -30, -35, -36, -38] # changes from RMP in response to subthreshold current step inputs
RMP = -67 #+/- 5mV 
# Make sure temp is 37C!!

## (2) DATA FROM MODEL
### Delete pre-existing data dir
if os.path.isdir('data'):
	shutil.rmtree('data')

### Run batch 
overlay_batch.currentSteps()

### Extract data from batch run
model_data = []
for i in range(13):
	data = json.load(open('/Users/ericagriffith/Desktop/NGF_tests/NGF_overlay/data/currentStep__' + str(i) + '.json'))
	vdata_total = data['simData']['V_soma']['cell_0']
	vdata = vdata_total[12500:62500]
	mean_responses = mean(vdata)
	model_data.append(mean_responses)

## RMP + RMP fitness
RMP_model = model_data[0]
RMP_fitness = abs(RMP_model - RMP)**2
print(RMP_fitness)

## SUBTHRESHOLD DELTAS + FITNESS
subthresh_deltas_model = []
for resp in model_data[1:]:
	delta = resp - RMP_model
	subthresh_deltas_model.append(delta)

compare = []
for i in range(len(subthreshold_deltas)):
	compare.append((subthreshold_deltas[i] - subthresh_deltas_model[i])**2)
subthresh_fitness = 0
for i in range(len(compare)):
	subthresh_fitness += compare[i]
print(subthresh_fitness)

## NOW PLOT DATA 
current_steps = [0,-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1, -0.11, -0.12]

plt.xlabel('Current Steps, nA')
plt.ylabel('Deflections from RMP, mV')
plt.axis([-0.15,0,-60,5])
plt.plot(current_steps[1:],subthreshold_deltas,'bs',label='Blue Square: Empirical')
plt.plot(current_steps[1:],subthresh_deltas_model,'g^',label='Green Triangle: Model\nRMP_fitness: ' + str(round(RMP_fitness)) + '\nsubthresh_fitness: ' + str(round(subthresh_fitness)))
plt.legend(loc='lower right')
plt.show()