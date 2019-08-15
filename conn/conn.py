# A1 Connectivity 

# Load data from published studies
def loadData():
    data = {'prob': 0.2, 'weight': 0.5}  # oversimplified example -- will have data from different papers for differnet projections 
    return data

# Params
Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']  # all layers

Ipops = ['NGF1',                            # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
        'PV6', 'SOM6', 'VIP6', 'NGF6']      # L6 

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
# Probabilities 

## E -> E 
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


## E -> I
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

## I -> E  
# (here illustrating a different way of populatinf the pmat dict)
for pop in Ipops:
    pmat[pop]['IT2'] = data['prob']
    pmat[pop]['IT3'] = data['prob']
    pmat[pop]['ITP4'] = data['prob']
    pmat[pop]['ITS4'] = data['prob']
    pmat[pop]['IT5A'] = data['prob']
    pmat[pop]['IT5B'] = data['prob']
    pmat[pop]['PT5B'] = data['prob']  
    pmat[pop]['IT6'] = data['prob']  
    pmat[pop]['CT6'] = data['prob']  

## I -> I
for pre in Ipops:
    for post in Ipops:
        pmat[pre][post] = data['prob']


# --------------------------------------------------
# Weights (=unitary conn somatic PSP amplitude)

## E -> E 
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


## E -> I
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

## I -> E  
# (here illustrating a different way of populatinf the wmat dict)
for pop in Ipops:
    wmat[pop]['IT2'] = data['weight']
    wmat[pop]['IT3'] = data['weight']
    wmat[pop]['ITP4'] = data['weight']
    wmat[pop]['ITS4'] = data['weight']
    wmat[pop]['IT5A'] = data['weight']
    wmat[pop]['IT5B'] = data['weight']
    wmat[pop]['PT5B'] = data['weight']  
    wmat[pop]['IT6'] = data['weight']  
    wmat[pop]['CT6'] = data['weight']  

## I -> I
for pre in Ipops:
    for post in Ipops:
        wmat[pre][post] = data['weight']


# --------------------------------------------------
# Delays
## Make distance-dependent for now


# --------------------------------------------------
# Save data to pkl file
savePickle = 1

if savePickle:
    import pickle
    with open('conn.pkl', 'wb') as f:
        pickle.dump({'pmat': pmat, 'wmat': wmat}, f)