# batch run 
from netpyne import specs
from netpyne.batch import Batch
import os
import sys

#runType='hpc_slurm'	# COMET
#runType = 'mpi_bulletin'	# LAPTOP or ZN
batchLabel='NGF_batch_data'

def batch_full(icand, ngen, runType):
    # Create variable of type ordered dictionary (NetPyNE's customized version) 
    params=specs.ODict()

    # Fill in with parameters to explore and range of values (key has to coincide with a variable in cfg) 
    params['amp'] = [0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1, -0.11, -0.12]

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, cfgFile='NGF_cfg.py', netParamsFile='NGF_netParams.py')

    # Set output folder, grid method (all param combinations), and run configuration
    b.batchLabel = batchLabel + '_cand' + str(icand)
    b.method = 'grid'

    if runType == 'hpc_slurm':
        b.saveFolder = '/oasis/scratch/comet/eyg42/temp_project/A1/NGF_passive/' + batchLabel  + '_gen_' + str(ngen)
        b.runCfg = {'type': 'hpc_slurm',
                                'allocation': 'shs100',
                                'walltime': '00:30:00',
                                'nodes': 1,
                                'coresPerNode': 1,
                                'email': 'ericaygriffith@gmail.com',
                                'folder': '/home/eyg42/A1/cells/NGF_optimization/NGF_passive',
                                'script': 'NGF_init.py',
                                'mpiCommand': 'ibrun',
                                'skip': False,
                                'custom': '#SBATCH --partition=shared'}
                                
    elif runType == 'hpc_slurm_gcp':
        b.saveFolder = '/home/salvadord/A1/cells/NGF_optimization/NGF_passive/' + batchLabel  + '_gen_' + str(ngen)
        b.runCfg = {'type': 'hpc_slurm', 
            'allocation': 'default', # bridges='ib4iflp', comet m1='shs100', comet nsg='csd403', gcp='default'
            'walltime': '00:10:00', #'48:00:00',
            'nodes': 1,
            'coresPerNode': 1,  # comet=24, bridges=28, gcp=32
            'email': 'salvadordura@gmail.com',
            'folder': '/home/salvadord/A1/cells/NGF_optimization/NGF_passive',  # comet,gcp='/salvadord', bridges='/salvi82'
            'script': 'NGF_init.py', 
            'mpiCommand': 'mpirun', # comet='ibrun', bridges,gcp='mpirun' 
            'skipCustom': '_raster.png',
            'custom': '#SBATCH --partition=shared'}
            #'custom': '#SBATCH --exclude=compute[17-64000]'} # only use first 16 nodes (non-preemptible for long runs )
            # --nodelist=compute1

    elif runType == 'mpi_bulletin':
        if not os.path.isdir(batchLabel + '_gen_' + str(ngen)):
            os.mkdir(batchLabel + '_gen_' + str(ngen))
        b.saveFolder = batchLabel + '_gen_' + str(ngen)
        b.runCfg = {'type': 'mpi_bulletin',
                        'script': 'NGF_init.py',
                        'skip': False}	# Needs to be false so can successfully run batches in optimization scripts -- is this true? 

    else:
        raise Exception("runType must be either 'hpc_slurm' or 'mpi_bulletin'.")


    # Run batch simulations
    b.run()

# Main code
if __name__ == '__main__':
    batch_full(sys.argv[1],sys.argv[2],sys.argv[3])



