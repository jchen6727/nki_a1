"""
dataAnalysis.py 

Code to anlayze non-human primate data 

Contributors: ericaygriffith@gmail.com, samnemo@gmail.com
"""


try:
    basestring
except NameError:
    basestring = str


## IMPORTS ## 
import sys
import os
import shutil 
import h5py									              # for rdmat() and getTriggerTimes()
import numpy as np
import downsample
from collections import OrderedDict
from filter import lowpass,bandpass 		  # for getbandpass()
import scipy                              # for plotCSD()
import matplotlib                         # for plotCSD()
from matplotlib import pyplot as plt      # for plotCSD() 


## PRE-PROCESSING FUNCTIONS ## 
### Originally in rdmat.py ### 
def rdmat (fn,samprds=0):  
  fp = h5py.File(fn,'r') # open the .mat / HDF5 formatted data
  sampr = fp['craw']['adrate'][0][0] # original sampling rate
  print('fn:',fn,'sampr:',sampr,'samprds:',samprds)
  dt = 1.0 / sampr # time-step in seconds
  dat = fp['craw']['cnt'] # cnt record stores the electrophys data
  npdat = np.zeros(dat.shape)
  tmax = ( len(npdat) - 1.0 ) * dt # use original sampling rate for tmax - otherwise shifts phase
  dat.read_direct(npdat) # read it into memory; note that this LFP data usually stored in microVolt
  npdat *= 0.001 # convert microVolt to milliVolt here
  fp.close()
  if samprds > 0.0: # resample the LFPs
    dsfctr = sampr/samprds
    dt = 1.0 / samprds
    siglen = max((npdat.shape[0],npdat.shape[1]))
    nchan = min((npdat.shape[0],npdat.shape[1]))
    npds = [] # zeros((int(siglen/float(dsfctr)),nchan))
    # print dsfctr, dt, siglen, nchan, samprds, ceil(int(siglen / float(dsfctr))), npds.shape
    for i in range(nchan): 
      print('resampling channel', i)
      npds.append(downsample.downsample(npdat[:,i], sampr, samprds))
    npdat = np.array(npds)
    npdat = npdat.T
    sampr = samprds
  tt = np.linspace(0,tmax,len(npdat)) # time in seconds
  return sampr,npdat,dt,tt # npdat is LFP in units of milliVolt


# bandpass filter the items in lfps. lfps is a list or numpy array of LFPs arranged spatially by column
### Originally in load.py ### 
def getbandpass (lfps,sampr,minf=0.05,maxf=300):
  datband = []
  for i in range(len(lfps[0])): datband.append(bandpass(lfps[:,i],minf,maxf,df=sampr,zerophase=True))
  datband = np.array(datband)
  return datband


# Vaknin correction for CSD analysis
# Allows CSD to be performed on all N contacts instead of N-2 contacts
# See Vaknin et al (1989) for more details
### Originally in load.py ### 
def Vaknin(x):
    # Preallocate array with 2 more rows than input array
    x_new = np.zeros((x.shape[0]+2, x.shape[1]))

    # Duplicate first and last row of x into first and last row of x_new
    x_new[0, :] = x[0, :]
    x_new[-1, :] = x[-1, :]

    # Duplicate all of x into middle rows of x_new
    x_new[1:-1, :] = x

    return x_new


# get CSD - first do a lowpass filter. lfps is a list or numpy array of LFPs arranged spatially by column
def getCSD (lfps,sampr,spacing_um,minf=0.05,maxf=300):

  # convert from uV to mV
  lfps = lfps/1000

  datband = getbandpass(lfps,sampr,minf,maxf)
  if datband.shape[0] > datband.shape[1]: # take CSD along smaller dimension
    ax = 1
  else:
    ax = 0

  spacing_mm = spacing_um/1000
  # when drawing CSD make sure that negative values (depolarizing intracellular current) drawn in red,
  # and positive values (hyperpolarizing intracellular current) drawn in blue
  CSD = -np.diff(datband,n=2,axis=ax)/spacing_mm**2 # now each column (or row) is an electrode -- CSD along electrodes

  CSD = Vaknin(CSD)

  return CSD


#
def getTriggerTimes (fn):
  fp = h5py.File(fn,'r')
  hdf5obj = fp['trig/anatrig']
  x = np.array(fp[hdf5obj.name])
  val = [y[0] for y in fp[x[0,0]].value]
  fp.close()
  return val  

#
def loadfile (fn,samprds,spacing_um=100):
  # load a .mat data file (fn) using sampling rate samprds (should be integer factor of original sampling rate (44000)),
  # returns: sampr is sampling rate after downsampling
  #          LFP is laminar local field potential data
  #          dt is time-step (redundant with sampr)
  #          tt is time array (in seconds)
  #          CSD is laminar current source density signal
  #          trigtimes is array of stimulus trigger indices (indices into arrays)
  sampr,LFP,dt,tt=rdmat(fn,samprds=samprds) # # samprds = 11000.0 # downsampling to this frequency
  sampr,dt,tt[0],tt[-1] # (2000.0, 0.001, 0.0, 1789.1610000000001)
  CSD = getCSD(LFP,sampr,spacing_um)
  divby = 44e3 / samprds
  trigtimes = None
  try: # not all files have stimuli
    trigtimes = [int(round(x)) for x in np.array(getTriggerTimes(fn)) / divby] # divby since downsampled signals by factor of divby
  except:
    pass
  #trigIDs = getTriggerIDs(fn)
  LFP = LFP.T # make sure each row is a channel
  return sampr,LFP,dt,tt,CSD,trigtimes


### AVERAGING FUNCTIONS ###
#### NOTE: should also make these available for use for sim data as well in netpyne 
def ms2index (ms, sampr): return int(sampr*ms/1e3)

# get the average ERP (dat should be either LFP or CSD)
# originally from load.py 
def getAvgERP (dat, sampr, trigtimes, swindowms, ewindowms):
  nrow = dat.shape[0]
  tt = np.linspace(swindowms, ewindowms,ms2index(ewindowms - swindowms,sampr))
  swindowidx = ms2index(swindowms,sampr) # could be negative
  ewindowidx = ms2index(ewindowms,sampr)
  avgERP = np.zeros((nrow,len(tt)))
  for chan in range(nrow): # go through channels
    for trigidx in trigtimes: # go through stimuli
      sidx = max(0,trigidx+swindowidx)
      eidx = min(dat.shape[1],trigidx+ewindowidx)
      avgERP[chan,:] += dat[chan, sidx:eidx]
    avgERP[chan,:] /= float(len(trigtimes))
  return tt,avgERP


### PLOTTING FUNCTIONS ### 
# PLOT CSD 
def plotCSD(dat,tt,fn=None,overlay=None,LFP_data=None,timeRange=None,saveFig=True,showFig=True):
  ## dat --> CSD data as numpy array
  ## timeRange --> time range to be plotted (in ms)
  ## tt --> numpy array of time points (time array in seconds)
  ## fn --> filename -- used for saving! 
  ## overlay --> can be 'LFP' or 'CSD' to overlay time series of either dataset 
  ## LFP_data --> numpy array of LFP data 

  
  tt = tt*1e3 # Convert units from seconds to ms 
  dt = tt[1] - tt[0] # dt is the time step of the recording # UNITS: in ms after converstion

  if timeRange is None:
    timeRange = [0,tt[-1]] # if timeRange is not specified, it takes the entire time range of the recording (ms)
  else:
    dat = dat[:,int(timeRange[0]/dt):int(timeRange[1]/dt)] # SLICE CSD DATA APPROPRIATELY
    tt = tt[int(timeRange[0]/dt):int(timeRange[1]/dt)] # DO THE SAME FOR TIME POINT ARRAY 

  # INTERPOLATION
  X = tt 
  Y = np.arange(dat.shape[0]) 
  CSD_spline = scipy.interpolate.RectBivariateSpline(Y,X,dat)
  Y_plot = np.linspace(0,dat.shape[0],num=1000) 
  Z = CSD_spline(Y_plot,X)

  # (i) Set up axes
  xmin = int(X[0])
  xmax = int(X[-1]) + 1 
  ymin = 1  # 0 in csd.py in netpyne 
  ymax = 24 # 24 in csd_verify.py, but it is spacing in microns in csd.py in netpyne --> WHAT TO DO HERE? TRY 24 FIRST! 
  extent_xy = [xmin, xmax, ymax, ymin]

  # (ii) Set up figure
  fig = plt.figure()

  # (iii) Create plots w/ common axis labels and tick marks
  axs = []

  numplots = 1 # HOW TO DETERMINE THIS? WHAT IF MORE THAN ONE? 

  gs_outer = matplotlib.gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.2, height_ratios = [20, 1])

  for i in range(numplots):
    axs.append(plt.Subplot(fig,gs_outer[i*2:i*2+2]))
    fig.add_subplot(axs[i])
    axs[i].set_xlabel('Time (ms)',fontsize=12)
    axs[i].tick_params(axis='y', which='major', labelsize=8)

  # (iv) PLOT INTERPOLATED CSD COLOR MAP
  spline=axs[0].imshow(Z, extent=extent_xy, interpolation='none', aspect='auto', origin='upper', cmap='jet_r', alpha=0.9) # alpha controls transparency -- set to 0 for transparent, 1 for opaque
  axs[0].set_ylabel('Channel', fontsize = 12) # Contact depth (um) -- convert this eventually 


  # (v) OVERLAY -- SETTING ASIDE FOR NOW -- THAT IS NEXT GOAL 
  if overlay is 'LFP' or overlay is 'CSD':
    nrow = dat.shape[0] # number of channels
    gs_inner = matplotlib.gridspec.GridSpecFromSubplotSpec(nrow, 1, subplot_spec=gs_outer[0:2], wspace=0.0, hspace=0.0) 
    subaxs = []

    # go down grid and add data from each channel 
    if overlay == 'LFP':
      if LFP_data is None:
        print('no LFP data provided!')
      else:
        axs[0].set_title('NHP CSD with LFP overlay', fontsize=14)
        LFP_data = LFP_data[:,int(timeRange[0]/dt):int(timeRange[1]/dt)] # slice LFP data according to timeRange
        for chan in range(nrow):
          subaxs.append(plt.Subplot(fig,gs_inner[chan],frameon=False))
          fig.add_subplot(subaxs[chan])
          subaxs[chan].margins(0.0,0.01)
          subaxs[chan].get_xaxis().set_visible(False)
          subaxs[chan].get_yaxis().set_visible(False)
          subaxs[chan].plot(X,LFP_data[chan,:],color='gray',linewidth=0.3)

    elif overlay == 'CSD':
      axs[0].set_title('NHP CSD with CSD time series overlay', fontsize=14)
      for chan in range(nrow):
          subaxs.append(plt.Subplot(fig,gs_inner[chan],frameon=False))
          fig.add_subplot(subaxs[chan])
          subaxs[chan].margins(0.0,0.01)
          subaxs[chan].get_xaxis().set_visible(False)
          subaxs[chan].get_yaxis().set_visible(False)
          subaxs[chan].plot(X,dat[chan,:],color='red',linewidth=0.3)

  else:
    axs[0].set_title('NHP Current Source Density (CSD)', fontsize=14)


  # SAVE FIGURE
  if saveFig:
    if fn is None:
      if overlay == 'LFP':
        figname = 'NHP_CSD_withLFP.png'
      elif overlay == 'CSD':
        figname = 'NHP_CSD_csdOverlay.png'
      else:
        figname = 'NHP_CSD_fig.png'
    else:
      filename = fn[31:-4] # takes out the .mat from the filename given as arg
      if overlay == 'LFP':
        figname = 'NHP_CSD_withLFP_%s.png' % filename
      elif overlay == 'CSD':
        figname = 'NHP_CSD_csdOverlay_%s.png' % filename
      else:
        figname = 'NHP_CSD_fig_%s.png' % filename
    try:
      plt.savefig(figname) # dpi
    except:
      plt.savefig('NHP_CSD_fig.png')



  # DISPLAY FINAL FIGURE
  if showFig is True:
    plt.show()
    #plt.close()



### PLOT CSD OF AVERAGED ERP ### 
def plotAvgCSD(dat,tt,fn=None,overlay=True,saveFig=True,showFig=True):
  ## dat --> CSD data as numpy array (from getAvgERP)
  ## tt --> numpy array of time points (from getAvgERP)
  ## fn --> string --> input filename --> used for saving! 
  ## overlay --> Default TRUE --> plots avgERP CSP time series on top of CSD color map 

  # INTERPOLATION
  X = tt 
  Y = np.arange(dat.shape[0]) # make sure this is the right axis ([0] correct for sim data) # may be [1] for data 
  CSD_spline = scipy.interpolate.RectBivariateSpline(Y,X,dat)
  Y_plot = np.linspace(0,dat.shape[0],num=1000) # ,num=1000 is included in csd.py in netpyne --> hmm. necessary? 
  Z = CSD_spline(Y_plot,X)

  # (i) Set up axes
  xmin = int(X[0])
  xmax = int(X[-1]) + 1 
  ymin = 1  # 0 in csd.py in netpyne 
  ymax = 24 # dat.shape[0] # 24 in csd_verify.py, but it is spacing in microns in csd.py in netpyne --> WHAT TO DO HERE? TRY 24 FIRST! 
  extent_xy = [xmin, xmax, ymax, ymin]

  # (ii) Set up figure
  fig = plt.figure()

  # (iii) Create plots w/ common axis labels and tick marks
  axs = []

  numplots = 1 # HOW TO DETERMINE THIS? WHAT IF MORE THAN ONE? 

  gs_outer = matplotlib.gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.2, height_ratios = [20, 1])

  for i in range(numplots):
    axs.append(plt.Subplot(fig,gs_outer[i*2:i*2+2]))
    fig.add_subplot(axs[i])
    axs[i].set_xlabel('Time (ms)',fontsize=12)
    axs[i].tick_params(axis='y', which='major', labelsize=8)

  # (iv) PLOT INTERPOLATED CSD COLOR MAP
  spline=axs[0].imshow(Z, extent=extent_xy, interpolation='none', aspect='auto', origin='upper', cmap='jet_r', alpha=0.9) # alpha controls transparency -- set to 0 for transparent, 1 for opaque
  axs[0].set_ylabel('Channel', fontsize = 12) # Contact depth (um) -- convert this eventually 


  # (v) SET TITLE AND OVERLAY AVERAGE ERP TIME SERIES (OR NOT)
  ## NOTE: add option for overlaying average LFP...??
  if overlay:
    nrow = dat.shape[0] # number of channels 
    gs_inner = matplotlib.gridspec.GridSpecFromSubplotSpec(nrow, 1, subplot_spec=gs_outer[0:2], wspace=0.0, hspace=0.0)
    subaxs = []

    # set title
    axs[0].set_title('NHP CSD with CSD time series overlay',fontsize=12)
    # go down grid and add data from each channel
    for chan in range(nrow):
        subaxs.append(plt.Subplot(fig,gs_inner[chan],frameon=False))
        fig.add_subplot(subaxs[chan])
        subaxs[chan].margins(0.0,0.01)
        subaxs[chan].get_xaxis().set_visible(False)
        subaxs[chan].get_yaxis().set_visible(False)
        subaxs[chan].plot(X,dat[chan,:],color='red',linewidth=0.3)

  else:
    axs[0].set_title('NHP Current Source Density (CSD)', fontsize=14)



  # SAVE FIGURE
  ## make this a little more explicable 
  if saveFig:
    if fn is None: 
      if overlay:
        figname = 'NHP_avgCSD_csdOverlay.png'
      else:
        figname = 'NHP_avgCSD.png'
    else:
      filename = fn[31:-4] # removes the .mat from the input filename
      print(filename)
      if overlay:  
        figname = 'NHP_avgCSD_csdOverlay_%s.png' % filename
      else:
        figname = 'NHP_avgCSD_%s.png' % filename
    try:
      plt.savefig(figname) #dpi
    except:
      plt.savefig('NHP_avgCSD.png')


  # DISPLAY FINAL FIGURE
  if showFig is True:
    plt.show()
    #plt.close()


##################################  
### FILE PROCESSING FUNCTIONS #### 
##################################

# What functions do I need? 
## Function that does 4 (removes or moves other .mat files)

def sortFiles(pathToData,regions):
  # pathToData -- string -- should go to parent directory with the raw unsorted .mat files
  # regions -- list of string or numbers -- either number code or name code for recording regions of interest (e.g. ['A1' 'MGB'] or [1 7])
  ## ^^ Make it so can either delete or sort the files not recorded in 'regions'

  # (1) Create a list of all the unsorted .mat files 
  ## NOTE: COMBINE THESE LINES? TEST. 
  origDataFiles = [f for f in os.listdir(pathToData) if os.path.isfile(os.path.join(pathToData,f))]
  origDataFiles = [f for f in origDataFiles if '.mat' in f] # list of all the .mat data files to be processed 
  print(origDataFiles)


  # (2) Set up dict to contain A1, MGB, and TRN filenames
  recordingAreaCodes = {1:'A1', 2:'belt', 3:'MGB', 4:'LGN', 5:'Medial Pulvinar', 6:'Pulvinar', 7:'TRN', 8:'Motor Ctx', 9:'Striatum', 10:'SC', 11:'IP', 33:'MGBv'} # All of the area codes -- recording region pairs 
  numCodes = list(recordingAreaCodes.keys()) # [1, 3, 7]
  nameCodes = list(recordingAreaCodes.values()) # ['A1', 'MGB', 'TRN']

  DataFiles = {} 

  for fn in origDataFiles:
    fullFN = pathToData + fn
    #print(fullFN)
    fp = h5py.File(fullFN,'r')
    areaCode = int(fp['params']['filedata']['area'][0][0]) # 1 - A1   # 3 - MGB   # 7 - TRN
    if areaCode in numCodes:
      area = str(recordingAreaCodes[areaCode]) # e.g. 'A1', 'MGB'
      if area not in list(DataFiles.keys()):
        DataFiles[area] = [] 
        DataFiles[area].append(fn)
      else:
        DataFiles[area].append(fn)
    else:
      print('Invalid area code in file %s' % fn)


  # (3) Move files into appropriate subdirectories
  for key in DataFiles.keys():
    for file in DataFiles[key]:
      origFilePath = pathToData + file
      newPath = pathToData + key + '/' # /A1/ etc. 
      newFilePath = newPath + file 
      if os.path.isdir(newPath):
        shutil.move(origFilePath,newFilePath)
      elif not os.path.isdir(newPath):
        os.mkdir(newPath)
        shutil.move(origFilePath,newFilePath)

  return DataFiles # Change this...? 


## REVAMP THIS 
def moveDataFiles(pathToData,option): # RENAME THIS ## deletes or moves irrelevant .mat files
  # pathToData -- path to parent dir with unsorted or unwanted .mat files 
  # option -- delete or move to 'other'

  # list of unsorted files 
  leftoverFiles = [q for q in os.listdir(pathToData) if os.path.isfile(os.path.join(pathToData,q))]
  leftoverFiles = [q for q in leftoverFiles if '.mat' in q]

  for left in leftoverFiles:
    fullLeft = pathToData + left             # full path to leftover .mat file 
    if option is None or option is 'delete':  # DEFAULT IS TO DELETE THE OTHER UNSORTED FILES
      if os.path.isfile(fullLeft):
        print('Deleting ' + left)             # INSTEAD OF DELETING SHOULD I JUST MOVE THE FILE? 
        os.remove(fullLeft)
    elif option is 'move': # MOVE TO 'other' DIRECTORY
      otherDir = pathToData + 'other/'
      otherFilePath = otherDir + left
      if os.path.isdir(otherDir):
        shutil.move(fullLeft,otherFilePath)
      elif not os.path.isdir(otherDir):
        os.mkdir(otherDir)
        shutil.move(fullLeft,otherFilePath)


###########################
######## MAIN CODE ########
###########################

if __name__ == '__main__':

  # Parent data directory containing unsorted .mat files
  origDataDir = '../data/NHPdata/click/contproc/'   #'/Users/ericagriffith/Documents/MATLAB/macaque_A1/click/contproc/'
  
  # Sort these files by recording region 
  DataFiles = sortFiles(origDataDir, [1, 3, 7]) # path to data .mat files  # recording regions of interest

  moveDataFiles(origDataDir,'move')

  # # (4) Delete the files that haven't been moved 
  # leftoverFiles = [q for q in os.listdir(origDataDir) if os.path.isfile(os.path.join(origDataDir,q))]
  # leftoverFiles = [q for q in leftoverFiles if '.mat' in q]

  # for left in leftoverFiles:
  #   fullLeft = origDataDir + left
  #   if os.path.isfile(fullLeft):
  #     print('Deleting ' + left)  # INSTEAD OF DELETING SHOULD I JUST MOVE THE FILE? 
  #     os.remove(fullLeft)


  # ## CSD PROCESSING
  # dataType = 'click' # 'spont' # 'speech'
  
  # if dataType == 'click': 
  #   paths_to_Data = ['../data/NHPdata/click/contproc/A1/', '../data/NHPdata/click/contproc/MGB/', '../data/NHPdata/click/contproc/TRN/']
  #   paths_to_Figs = ['../data/NHPdata/CSD/click/A1/', '../data/NHPdata/CSD/click/MGB/', '../data/NHPdata/CSD/click/TRN/']
  # elif dataType == 'spont':
  #   paths_to_Data = ['../data/NHPdata/spont/contproc/A1/', '../data/NHPdata/spont/contproc/MGB/', '../data/NHPdata/spont/contproc/TRN/']
  #   paths_to_Figs = ['../data/NHPdata/CSD/spont/A1/', '../data/NHPdata/CSD/spont/MGB/', '../data/NHPdata/CSD/spont/TRN/']    
  # elif dataType == 'speech':
  #   paths_to_Data = ['../data/NHPdata/speech/contproc/A1/', '../data/NHPdata/speech/contproc/MGB/', '../data/NHPdata/speech/contproc/TRN/']
  #   paths_to_Figs = ['../data/NHPdata/CSD/speech/A1/', '../data/NHPdata/CSD/speech/MGB/', '../data/NHPdata/CSD/speech/TRN/']

  # for pathData in paths_to_Data:
  #   #os.listdir(pathData)
  #   #print('Processing .mat files from ' + pathData)

  #   dataFiles = [f for f in os.listdir(pathData) if os.path.isfile(os.path.join(pathData,f))]

  #   dataFiles = [f for f in dataFiles if '.mat' in f] # gets all the dataFiles in either A1, MGB, or TRN subdirs 

  #   #print(dataFiles)

  #   for file in dataFiles:

  #     filepath = pathData + file # string with full path to file 

  #     [sampr,LFP_data,dt,tt,CSD_data,trigtimes] = loadfile(fn=filepath, samprds=11*1e3, spacing_um=100)
  #     # sampr is the sampling rate after downsampling 
  #     # tt is time array (in seconds)
  #     # trigtimes is array of stim trigger indices

  #     #### PLOT INTERPOLATED CSD COLOR MAP PLOT #### 
  #     plotCSD(fn=filepath,dat=CSD_data,tt=tt,timeRange=[1100,1200],showFig=False)


  #     # REMOVE BAD EPOCHS FIRST..?  

  #     # GET AVERAGE ERP ## 
  #     ## set epoch params
  #     swindowms = 0 # start time relative to stimulus 
  #     ewindowms = 200 # end time of epoch relative to stimulus onset 

  #     # calculate average CSD ERP 
  #     ttavg,avgCSD = d.getAvgERP(CSD_data, sampr, trigtimes, swindowms, ewindowms)
  #     plotAvgCSD(fn=filepath,dat=avgCSD,tt=ttavg,showFig=False)


    # # MOVE .PNG FILES 
    # pngFiles = [f for f in os.listdir() if os.path.isfile(f)]
    # pngFiles = [f for f in pngFiles if '.png' in f]
    # dataPrefixes = []



