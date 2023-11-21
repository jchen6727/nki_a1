import pickle
import numpy as np

sim = pickle.load(open('data/optunaERP_23nov13_/gen_45/trial_45_data.pkl', 'rb'))   # or place the trial_45_data.pkl here

d = pickle.load(open('opt/2-rb023024011@os.mat_20kHz_avgERP.pkl','rb'))
ttavgERPNHP = d['ttavg']
avgCSDNHP = d['avgCSD'] # s2, g, i1 channels for primary CSD sinks are at indices 10, 14, 19
fitnessFuncArgs = {}
fitnessFuncArgs['maxFitness'] = 1.0
groupedParams = []

def fitnessFunc(simData, **kwargs):
    print('fitness func')
    from csd import getCSDa1dat as getCSD
    from scipy.stats import pearsonr
    from erp import getAvgERP
    def ms2index (ms, sampr): return int(sampr*ms/1e3)
    LFP = simData['LFP']
    LFP = np.array(LFP)
    CSD = getCSD(LFP, 1e3/0.05)
    CSD.shape # (18, 220000)
    lchan = [4, 10, 15]
    lnhpchan = [11-1, 15-1, 20-1]
    bbnT = np.arange(3000, 4000, 300)
    dt = 0.05
    sampr = 1e3/dt
    bbnTrigIdx = [ms2index(x,sampr) for x in bbnT]
    ttERP,avgERP = getAvgERP(CSD, sampr, bbnTrigIdx, 0, 150)
    fitness = -pearsonr(avgCSDNHP[lnhpchan[1],:],avgERP[lchan[1]])[0]
    print('fitness is', fitness)
    return fitness

fitnessFunc(sim['simData'])