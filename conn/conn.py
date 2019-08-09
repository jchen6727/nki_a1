# A1 Connectivity 

# Params
Epops = ['IT2', 'IT4']
Ipops = ['IT2', 'IT4']

pmat = {}
for p in Epops+Ipops: pmat[p] = {}

wmat = {}
for p in Epops+Ipops: wmat[p] = {}


# Probabilities 

## E -> E
pmat['IT2']['IT2'] = 0.2

## E -> I

## I -> E

## I -> I

# Weights (=unitary conn somatic PSP amplitude)

## E -> E
wmat['IT2']['IT2'] = 0.5

## E -> I

## I -> E

## I -> I
