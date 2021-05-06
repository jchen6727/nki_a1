# script to print best fitness values
# printBest.py 

import sys
import re

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# read screenlog.0
try:
    with open('screenlog.0', 'r') as f:
        data = f.read()
        match = re.findall(r'Best is trial \d+ with value: \d+\.\d\d', data)

        matchUnique = unique(match)
        matchUnique = [m.replace('Best is t', '   T').replace(' with value:',':\t') for m in matchUnique]
        print('')
        for m in matchUnique: print(m) #pprint(matchUnique)
        print('')
        trials = re.findall(r'Trial \d+ finished with value', data)
        lastTrial = trials[-1].replace('Trial', '').replace('finished with value')
        print('Last trial: ')
        print('')
except:
    print ('Could not find screenlog.0')
    

