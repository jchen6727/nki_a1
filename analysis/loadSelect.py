
### Oscillation Event peakF calculations!! 

# IMPORTS # 
from evstats import *



noiseampCSD = 200.0 / 10.0 # amplitude cutoff for CSD noise; was 200 before units fix

###################################################
######### FUNCTIONS FOR getIEIstatsbyBand #########
###################################################

# get morlet specgrams on windows of dat time series (window size in samples = winsz)
def getmorletwin (dat,winsz,sampr,freqmin=1.0,freqmax=100.0,freqstep=1.0,\
                  noiseamp=noiseampCSD,getphase=False,useloglfreq=False,mspecwidth=7.0):
  lms = []
  n,sz = len(dat),len(dat)
  lnoise = []; lsidx = []; leidx = []
  if useloglfreq:
    minstep=0.1
    loglfreq = getloglfreq(freqmin,freqmax,minstep)  
  for sidx in range(0,sz,winsz):
    lsidx.append(sidx)
    eidx = sidx + winsz
    if eidx >= sz: eidx = sz - 1
    leidx.append(eidx)
    print(sidx,eidx)
    sig = dat[sidx:eidx]
    lnoise.append(max(abs(sig)) > noiseamp)
    if useloglfreq:
      ms = MorletSpec(sig,sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,lfreq=loglfreq,width=mspecwidth)
    else:
      ms = MorletSpec(sig,sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,width=mspecwidth)
    lms.append(ms)
  print('exiting getmorletwin-- returning values')
  return lms,lnoise,lsidx,leidx

#
def getDynamicThresh (lmsn, lnoise, thfctr, defthresh):
  from scipy.stats import chi2
  #lthresh = [np.percentile(chi2.pdf(x,2),95) for x,n in zip(lmsn,lnoise) if not n]
  lthresh = [mean(x)+thfctr*std(x) for x,n in zip(lmsn,lnoise) if not n]
  #lthresh = [np.percentile(x,95) for x,n in zip(lmsn,lnoise) if not n]
  if len(lthresh) > 0:
    print('Mean/min/max:',mean(lthresh),min(lthresh),max(lthresh))
    return min(lthresh)
  return defthresh # default is 4.0


# get oscillatory events
# lms is list of windowed morlet spectrograms, lmsnorm is spectrograms normalized by median in each power
# lnoise is whether the window had noise, medthresh is median threshold for significant events,
# lsidx,leidx are starting/ending indices into original time-series, csd is current source density
# on the single chan, MUA is multi-channel multiunit activity, overlapth is threshold for merging
# events when bounding boxes overlap, fctr is fraction of event amplitude to search left/right/up/down
# when terminating events
def getspecevents (lms,lmsnorm,lnoise,medthresh,lsidx,leidx,csd,MUA,chan,sampr,overlapth=0.5,endfctr=0.5,getphase=False):
  llevent = []
  for windowidx,offidx,ms,msn,noise in zip(arange(len(lms)),lsidx,lms,lmsnorm,lnoise): 
    imgpk = detectpeaks(msn) # detect the 2D local maxima
    print('imgpk detected')
    lblob = getblobsfrompeaks(msn,imgpk,ms.TFR,medthresh,endfctr=endfctr,T=ms.t,F=ms.f) # cut out the blobs/events
    print('lblob gotten')
    lblobsig = [blob for blob in lblob if blob.maxval >= medthresh] # take only significant events
    #print('ndups in lblobsig 0 = ', countdups(lblobsig), 'out of ', len(lblobsig))    
    lmergeset,bmerged = getmergesets(lblobsig,overlapth,areaop=min) # determine overlapping events
    lmergedblobs = getmergedblobs(lblobsig,lmergeset,bmerged)
    #print('ndups in lmergedblobs A = ', countdups(lmergedblobs), 'out of ', len(lmergedblobs))
    lmergeset,bmerged = getmergesets(lmergedblobs,1.0,areaop=max) # gets rid of duplicates
    lmergedblobs = getmergedblobs(lmergedblobs,lmergeset,bmerged)
    #print('ndups in lmergedblobs B = ', countdups(lmergedblobs), 'out of ', len(lmergedblobs))
    # get the extra features (before/during/after with MUA,avg,etc.)
    getextrafeatures(lmergedblobs,ms,msn,medthresh,csd,MUA,chan,offidx,sampr,endfctr=endfctr,getphase=getphase)
    print('extra features gotten')
    ndup = countdups(lmergedblobs)
    if ndup > 0: print('ndup in lmergedblobs = ', ndup, 'out of ', len(lmergedblobs))
    for blob in lmergedblobs: # store offsets for getting to time-series / wavelet spectrograms
      blob.windowidx = windowidx
      blob.offidx = offidx
      blob.duringnoise = noise
    llevent.append(lmergedblobs) # save merged events
    print('one iteration of getspecevents for-loop complete')
  return llevent

# get interevent interval distribution
def getblobIEI (lblob,scalex=1.0):
  liei = []
  newlist = sorted(lblob, key=lambda x: x.left)
  for i in range(1,len(newlist),1):
    liei.append((newlist[i].left-newlist[i-1].right)*scalex)
  return liei

# get event blobs in (inclusive for lower bound, strictly less than for upper bound) range of minf,maxf
def getblobinrange (lblobf, minF,maxF): return [blob for blob in lblobf if blob.peakF >= minF and blob.peakF < maxF]


# median normalization
def mednorm (dat,byRow=True):
  nrow,ncol = dat.shape[0],dat.shape[1]
  out = zeros((nrow,ncol))
  if byRow:
    for row in range(nrow):
      med = median(dat[row,:])
      if med != 0.0:
        out[row,:] = dat[row,:] / med
      else:
        out[row,:] = dat[row,:]
  else:
    for col in range(ncol):
      med = median(dat[:,col])
      if med != 0.0:
        out[:,col] = dat[:,col] / med
      else:
        out[:,col] = dat[:,col]
  return out

######################################################################################################
######################################################################################################

## getCV2, getLV, getFF?  -> not causing any problems for now. AH -- these come from evstats.py in a1dat

#
def getIEIstatsbyBand (dat,winsz,sampr,freqmin,freqmax,freqstep,medthresh,lchan,MUA,overlapth=0.5,getphase=True,savespec=False,useDynThresh=False,threshfctr=2.0,useloglfreq=False,mspecwidth=7.0,noiseamp=noiseampCSD,endfctr=0.5,normop=mednorm):
  # get the interevent statistics split up by frequency band
  dout = {'sampr':sampr,'medthresh':medthresh,'winsz':winsz,'freqmin':freqmin,'freqmax':freqmax,'freqstep':freqstep,'overlapth':overlapth}
  dout['threshfctr'] = threshfctr; dout['useDynThresh']=useDynThresh; dout['mspecwidth'] = mspecwidth; dout['noiseamp']=noiseamp
  dout['endfctr'] = endfctr
  for chan in lchan:
    dout[chan] = doutC = {'delta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'theta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'alpha':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'beta':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'gamma':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'hgamma':{'LV':[],'CV':[],'Count':[],'FF':None,'levent':[],'IEI':[]},
                          'lnoise':[]}
    print('up to channel', chan,'getphase:',getphase)
    if dat.shape[0] > dat.shape[1]:
      sig = dat[:,chan] # signal (either CSD or LFP)
      lms,lnoise,lsidx,leidx = getmorletwin(dat[:,chan],int(winsz*sampr),sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,useloglfreq=useloglfreq,mspecwidth=mspecwidth,noiseamp=noiseamp)
    else:
      sig = dat[chan,:] # signal (either CSD or LFP)
      lms,lnoise,lsidx,leidx = getmorletwin(dat[chan,:],int(winsz*sampr),sampr,freqmin=freqmin,freqmax=freqmax,freqstep=freqstep,getphase=getphase,useloglfreq=useloglfreq,mspecwidth=mspecwidth,noiseamp=noiseamp)
    print('completed morlet in getIEIstatsbyBand')
    if 'lsidx' not in dout: dout['lsidx'] = lsidx # save starting indices into original data array
    if 'leidx' not in dout: dout['leidx'] = leidx # save ending indices into original data array
    lmsnorm = [normop(ms.TFR) for ms in lms] # normalize wavelet specgram by median (when normop==mednorm) or unitnorm (sub avg div std)
    print('done with lmsnorm line')
    if useDynThresh: # using dynamic threshold?
      evthresh = getDynamicThresh(lmsnorm, lnoise, threshfctr, medthresh)
      print('useDynThresh=True, evthresh=',evthresh)
    else: #  otherwise use the default medthresh
      evthresh = medthresh
    print('evthresh = ' + str(evthresh))
    doutC['evthresh'] = evthresh # save the threshold used    
    print('doutC line completed')
    specsamp = lms[0].TFR.shape[1] # number of samples in spectrogram time axis
    print('specsamp line completed')
    specdur = specsamp / sampr # spectrogram duration in seconds
    print('specdur = ' + str(specdur))
    if 'specsamp' not in dout: dout['specsamp'] = specsamp
    if 'specdur' not in dout: dout['specdur'] = specdur    
    print ('2 if statements on specsamp and specsdur completed')
    llevent = getspecevents(lms,lmsnorm,lnoise,evthresh,lsidx,leidx,sig,MUA,chan,sampr,overlapth=overlapth,getphase=getphase,endfctr=endfctr) # get the spectral events
    print('completed llevent getspecevents')
    scalex = 1e3*specdur/specsamp # to scale indices to times
    print('scalex = ' + str(scalex))
    if 'scalex' not in dout: dout['scalex'] = scalex
    doutC['lnoise'] = lnoise # this is per channel - diff noise on each channel
    myt = 0
    for levent,msn,ms in zip(llevent,lmsnorm,lms):
      print(myt)
      """ do not skip noise so can look at noise event waveforms in eventviewer; can always filter out noise from dframe
      if lnoise[myt]: # skip noise
        myt+=1
        continue      
      """
      for band in dbands.keys(): # check events by band
        lband = getblobinrange(levent,dbands[band][0],dbands[band][1])
        count = len(lband)
        doutC[band]['Count'].append(count)
        doutC[band]['levent'].append(lband)
        if count > 2:
          lbandIEI = getblobIEI(lband,scalex)
          cv = getCV2(lbandIEI)
          doutC[band]['CV'].append(cv)
          doutC[band]['IEI'].append(lbandIEI)
        else:
          doutC[band]['IEI'].append([])
        if count > 3:
          lv = getLV(lbandIEI)
          doutC[band]['LV'].append(lv)
          print(band,len(lband),lv,cv)
      myt+=1
      for band in dbands.keys(): doutC[band]['FF'] = getFF(doutC[band]['Count'])
    if savespec:
      for MS,MSN in zip(lms,lmsnorm): MS.TFR = MSN # do not save lmsnorm separately, just copy it over to lms
      doutC['lms'] = lms
    else:
      del lms,lmsnorm # cleanup memory
      gc.collect()
  dout['lchan'] = lchan
  return dout

