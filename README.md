# Epidemics Simulation

Simulation of disease epidemics, on small world networks. 

The simulation accounts for different infection rates, different synergy parameters (collaborative or competitive diseases), and different network sizes.


## Installation

- Download this repository
- Install a virtual python environment (preferred python version is 3.7):

```bash
python -m venv venv
source venv/bin/activate
```
- Install dependencies using `pip install -r requirements.txt`


## Running a single simulation

Simulation are run using the `Network` classes. 

For instance, a rectangular network:

```python
from src.world.network.rectilinear import Rectilinear2DNetwork

my_simulation = Rectilinear2DNetwork(shape=(20, 20))
my_simulation.start_infection_at_center()
my_simulation.run()
```

This runs a single simulation, but does not output or save any results.

### Creating videos

A video of the simulation can be created by using the `SnapshotHistoryCallback` class:

```python

callback = SnapshotHistoryCallback()

my_simulation = Rectilinear2DNetwork(shape=(20, 20))
my_simulation.start_infection_at_center()
my_simulation.run(callback=callback)

callback.make_video(filename="simulation.mp4")
```

This will run a simulation and save a video visualation of the map over the duration of the simulation.


### Using MLFlow to save results

Simulations cab be tracked using the `MLFlowCallback` class:

```python
from src.world.network.rectilinear import Rectilinear2DNetwork

callback = MLFlowCallback(log_histories=False)

my_simulation = Rectilinear2DNetwork(shape=(20, 20))
my_simulation.start_infection_at_center()
my_simulation.run(callback=callback)
```

This will save the parameters and results of the simulation. Parameters are input choices, such as alpha (the synergy-free infection rate), beta (the synergy parameter), and network size. Output results include the percentage of cells infected at end; the final epidemic state of the simulation; etc.

The `MLFlowCallback` class has optional flag `log_histories`, which can be used to save the output parameters at all time steps in the simulation.


### Viewing results from MFLow

MFFlow results can be viewed using `mlflow ui` and opening your browser to the http:// address specified in the terminal output. The MLFlow UI displays each simulations as a single row, with input parameters and output parameters displayed as columns.


## Running many simulations

Large sets of simulations are most easily run using the script `mlflow_scripts/run_fill.py`. This script runs a large number of simulations, defined according to the config file (default `config.yml`). 

The default `config.yml` file runs at most 100 simulations (`maximum_runs`) with ranges of parameter values (`metrics.ALPHA` etc). The simulations are run in batches of size 50 (`runs_per_batch`) and saved to MLFlow database under the experiment name `synergy`.

The script `mlflow_scripts/visualise.py` can also be used to visualise how many simulations have been run for each set of parameters.
