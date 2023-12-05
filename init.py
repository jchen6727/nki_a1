"""
init.py

Starting script to run NetPyNE-based A1 model.


Usage:
    python init.py # Run simulation, optionally plot a raster


MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py


Contributors: ericaygriffith@gmail.com, salvadordura@gmail.com
"""

from neuron import h
onServer = False

from netpyne import sim

cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')

# sim.createSimulateAnalyze(netParams, cfg)
####### UNCOMMENTED THE ABOVE FOR TEST TRIAL RUN ON NEUROSIM 

sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  				# create network object and set cfg and net params
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)
alltags = sim._gatherAllCellTags()                      # gather cell tags; see https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/sim/gather.py
sim.runSim()                      			# run parallel Neuron simulation  

# sim.gatherData()                  			# gather spiking data and cell info from each node
# sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#

# if sim.rank==0: sim.analysis.plotData()         	# plot spike raster etc

# distributed saving (to avoid errors with large output data)
sim.saveDataInNodes()
sim.gatherDataFromFiles()

sim.saveData()  
####### COMMENTED OUT THE ABOVE FOR TEST TRIAL RUN ON NEUROSIM 

if sim.rank == 0:
    h.quit()