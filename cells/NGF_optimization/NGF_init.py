from netpyne import sim
# read cfg and netParams from command line arguments if available; otherwise use default
simConfig,netParams=sim.readCmdLineArgs(simConfigDefault='NGF_cfg.py',netParamsDefault='NGF_netParams.py')
# Create network and run simulation
sim.createSimulateAnalyze(netParams=netParams,simConfig=simConfig)