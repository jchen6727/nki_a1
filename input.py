# input.py
from brian2 import *
from brian2hears import *

def cochlearInputSpikes(freqRange=[9000,11000], # orig: [20, 20000],
                        numCells=200, # orig: [3000]
                        duration=1000,
                        toneFreq=10000,
                        plotRaster=False): 

    print(' Generating cochlear-like auditory input spikes using Brian Hears ...')
    cfmin, cfmax, cfN = freqRange[0]*Hz, freqRange[1]*Hz, numCells
    cf = erbspace(cfmin, cfmax, cfN)
    sound1 = tone(toneFreq*Hz, duration*ms)
    sound2 = whitenoise(duration*ms)
    sound = sound1+sound2
    sound = sound.ramp()
    gfb = Gammatone(sound, cf)
    ihc = FunctionFilterbank(gfb, lambda x: 3 * clip(x, 0, Inf)**(1.0 / 3.0))

    # Leaky integrate-and-fire model with noise and refractoriness
    eqs = '''
    dv/dt = (I-v)/(1*ms)+0.2*xi*(2/(1*ms))**.5 : 1 (unless refractory)
    I : 1
    '''
    G = FilterbankGroup(ihc, 'I', eqs, reset='v=0', threshold='v>1', refractory=5*ms, method='euler')

    # Run, and raster plot of the spikes
    M = SpikeMonitor(G)
    run(sound.duration)
    if plotRaster:
        plot(M.t / ms, M.i, '.')
        plt.show()

    # generate list of spk times
    spkts = list(M.t)
    spkids = list(M.i)
    spkTimes = [[] for i in range(numCells)] 
    for spkt, spkid in zip(spkts, spkids):
        spkTimes[spkid].append(float(spkt)*1000.)

    return spkTimes


# main
# auditoryInputSpikes(plotRaster=1)