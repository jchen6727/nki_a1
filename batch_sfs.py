import ray
import pandas
import json
import os
import numpy
import time

from ray import tune
from ray import air
from ray.air import session

from ray.tune.search import create_searcher, ConcurrencyLimiter, SEARCH_ALG_IMPORT
from ray.tune.search.basic_variant import BasicVariantGenerator
from pubtk.runtk.dispatchers import SFS_Dispatcher
from pubtk.runtk.submit import SGESubmitSFS

CONCURRENCY = 9
SAVESTR = 'grid.csv'

submit = SGESubmitSFS()

cwd = os.getcwd()

submit.update_template(
    command = "python netParams.py",
    cores = "5",
    vmem = "32G"
)

search_params = {
    # these params control IC -> Thal [LB, UB]
    'cfg.ICThalweightECore': tune.grid_search([0.75, 1.25]),
    'cfg.ICThalweightICore': tune.grid_search([0.1875, 0.3125]),
    'cfg.ICThalprobECore': tune.grid_search([0.1425, 0.2375]),
    'cfg.ICThalprobICore': tune.grid_search([0.09, 0.15]),
    'cfg.ICThalMatrixCoreFactor': tune.grid_search([0.075, 0.125]),
    # these params added from Christoph Metzner branch
    #'cfg.thalL4PV': tune.grid_search([0.1875, 0.3125]),
    #'cfg.thalL4SOM': tune.grid_search([0.1875, 0.3125]),
    #'cfg.thalL4E': tune.grid_search([1.5, 2.5])
}

def sge_run(config):
    gid = tune.get_trial_id()
    dispatcher = SFS_Dispatcher(cwd = cwd, env = {}, submit = submit, gid = gid)
    dispatcher.add_dict(value_type="FLOAT", dictionary = config)
    dispatcher.run()
    data = dispatcher.get_run()
    while not data:
        data = dispatcher.get_run()
        time.sleep(5)
    #dispatcher.clean(args='sw')
    data = pandas.read_json(data, typ='series', dtype=float)
    loss = data['FLOATRUNTK4'][3]
    session.report({'loss': loss, 'data': data})

algo = BasicVariantGenerator(max_concurrent=CONCURRENCY)

print("=====grid search=====")
print(search_params)

tuner = tune.Tuner(
    #objective,
    sge_run,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=1, # grid search samples 1 for each param
        metric="loss"
    ),
    run_config=air.RunConfig(
        local_dir="../ray_ses",
        name="grid",
    ),
    param_space=search_params,
)

results = tuner.fit()

resultsdf = results.get_dataframe()

resultsdf.to_csv(SAVESTR)

