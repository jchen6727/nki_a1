'''
Macaque A1 model 
Local connectivity preprocessing script

- Loads experimental data from file
- Preprocesses so ready to use by NetPyNE high-level specs

'''
import numpy as np

# Load data from published studies
def loadData():
    data = {'prob': 0.2, 'weight': 0.5}  # oversimplified example -- will have data from different papers for differnet projections 
    return data

# Params
Etypes = ['IT', 'ITS4', 'PT', 'CT']
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']  # all layers

Itypes = ['PV', 'SOM', 'VIP', 'NGF']
Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']  # L6 
        
layer = {'1': [0.00, 0.05], '2': [0.05, 0.08], '3': [0.08, 0.475], '4': [0.475,0.625], '5A': [0.625,0.667], '5B': [0.667,0.775], '6': [0.775,1]} # 

# initialize prob and weight matrices
# format: pmat[presynaptic_pop][postsynaptic_pop] 
pmat = {}
for p in Epops+Ipops: pmat[p] = {}  

wmat = {}
for p in Epops+Ipops: wmat[p] = {}

# Load exp data
data = loadData()

# Note: below shows just a simplified example of how the pmat and wmat dicts can be filled in.
# When using real exp it wil be much more complicated, require dividing by groups of pops, exceptions etc 


# --------------------------------------------------
## E -> E 
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 
for pop in Epops:
    pmat['IT2'][pop] = data['prob']  
    pmat['IT3'][pop] = data['prob']  
    pmat['ITP4'][pop] = data['prob']  
    pmat['ITS4'][pop] = data['prob']  
    pmat['IT5A'][pop] = data['prob']  
    pmat['IT5B'][pop] = data['prob']  
    pmat['PT5B'][pop] = data['prob']  
    pmat['IT6'][pop] = data['prob']  
    pmat['CT6'][pop] = data['prob']  

# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
for pop in Epops:
    wmat['IT2'][pop] = data['weight']  
    wmat['IT3'][pop] = data['weight']  
    wmat['ITP4'][pop] = data['weight']  
    wmat['ITS4'][pop] = data['weight']  
    wmat['IT5A'][pop] = data['weight']  
    wmat['IT5B'][pop] = data['weight']  
    wmat['PT5B'][pop] = data['weight']  
    wmat['IT6'][pop] = data['weight']  
    wmat['CT6'][pop] = data['weight']  

# --------------------------------------------------
## E -> I
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 
for pop in Ipops:
    pmat['IT2'][pop] = data['prob']  
    pmat['IT3'][pop] = data['prob']  
    pmat['ITP4'][pop] = data['prob']  
    pmat['ITS4'][pop] = data['prob']  
    pmat['IT5A'][pop] = data['prob']  
    pmat['IT5B'][pop] = data['prob']  
    pmat['PT5B'][pop] = data['prob']  
    pmat['IT6'][pop] = data['prob']  
    pmat['CT6'][pop] = data['prob']  


# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
for pop in Ipops:
    wmat['IT2'][pop] = data['weight']  
    wmat['IT3'][pop] = data['weight']  
    wmat['ITP4'][pop] = data['weight']  
    wmat['ITS4'][pop] = data['weight']  
    wmat['IT5A'][pop] = data['weight']  
    wmat['IT5B'][pop] = data['weight']  
    wmat['PT5B'][pop] = data['weight']  
    wmat['IT6'][pop] = data['weight']  
    wmat['CT6'][pop] = data['weight']
    

# --------------------------------------------------
## I -> E
# --------------------------------------------------

bins = {}
bins['inh'] = [[0.0, 0.37], [0.37, 0.8], [0.8,1.0]]

# --------------------------------------------------
## Probabilities 

'''
- I->E/I connections (Tremblay, 2016: Sohn, 2016; Budinger, 2018; Naka et al 2016; Garcia, 2015)
- Local, intralaminar only; all-to-all but distance-based; high weights
- Although evidence for L2/3,4,6 -> L5A/B, strongest is intralaminar (Naka16)
'''

# I->E particularities:
## inh cells target apical dends (not strictly local to layer) 
## L1 + L2/3 NGF -> L2/3 E (prox apic) + L5 E (tuft) -- specify target dend in subConnParams
## upper layer SOM, VIP, NGF project strongly to deeper E cells (Kato 2017) with exp-decay distance-dep

for pre in ['SOM', 'VIP', 'NGF']:
    pmat[(pre, 'E')] = np.array([[1.0, 1.0, 1.0],  # from L1+L2/3+L4 to all layers 
                                 [0.25, 1.0, 1.0],  # from L5A+L5B to all layers
                                 [0.25, 0.25, 1.0]])  # from L6 to all layers

## upper layer VP project weakly to deeper E cells (mostly intralaminar) (Kato 2017) with exp-decay distance-dep
## although Naka 2016 shows L2/3 PV -> L5 Pyr
pmat[('PV', 'E')] = np.array([[1.0, 0.5, 0.25],  # from L1+L2/3+L4 to all layers 
                              [0.25, 1.0, 0.5],  # from L5A+L5B to all layers
                              [0.1, 0.25, 1.0]])  # from L6 to all layers

# VIP -> E (very low; 3/42; Pi et al 2013) 
pmat[('VIP', 'E')] *= 0.1

# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
IEweight = 0.5
for pre in Ipops:
    for post in Ipops:
        wmat[pre][post] = IEweight


# --------------------------------------------------
## I -> I
# --------------------------------------------------

# --------------------------------------------------
## Probabilities 

### I->I particularities
for pre in Itypes:
    for post in Itypes:
        pmat[(pre, post)] = 1.0

# NGF -> all I, local/intralaminar (+ distance-dep)
# no change required

strong = 1.0
weak = 0.35  # = ~ normalized strong/weak = (4/15) / (14/18)
veryweak = 0.1

# VIP -> SOM (strong; 14/18), PV (weak; 4/15), VIP (very weak -- remove?) (Pi 2013)
pmat[('VIP', 'SOM')] = strong
pmat[('VIP', 'PV')] = weak
pmat[('VIP', 'VIP')] = veryweak
pmat[('VIP', 'NGF')] = weak  # unknown; assume weak

# SOM -> FS+VIP (strong); SOM (weak -- remove?)  (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
pmat[('SOM', 'PV')] = strong
pmat[('SOM', 'VIP')] = strong
pmat[('SOM', 'SOM')] = weak
pmat[('VIP', 'NGF')] = weak  # unknown; assume weak
 
# PV  -> PV (strong); SOM+VIP (weak -- remove?) (Naka et al 2016;Tremblay, 2016; Sohn, 2016)
pmat[('PV', 'PV')] = strong
pmat[('PV', 'SOM')] = weak
pmat[('PV', 'VIP')] = weak
pmat[('PV', 'NGF')] = weak  # unknown; assume weak


# --------------------------------------------------
## Weights  (=unitary conn somatic PSP amplitude)
IIweight = 0.5
for pre in Ipops:
    for post in Ipops:
        wmat[pre][post] = IIweight


# --------------------------------------------------
# Delays
## Make distance-dependent for now


# --------------------------------------------------
# Save data to pkl file
savePickle = 1

if savePickle:
    import pickle
    with open('conn.pkl', 'wb') as f:
        pickle.dump({'pmat': pmat, 'wmat': wmat, 'bins': bins}, f)