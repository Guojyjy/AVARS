# AVARS - Alleviating Unexpected Urban Road Traffic Congestion using UAVs

<div align=center>
    <img src='https://github.com/Guojyjy/AVARS/assets/109638662/4039f49f-713f-4568-8c0a-d169170cc5a3' width=80%>
    <div align=center> <strong>AVARS system flow</strong> </div>
</div>

<br>

The experiments are conducted on a simulator platform [SUMO](https://www.eclipse.org/sumo/). The model design and implementation are based on [Flow](https://flow-project.github.io), which configures with [RLlib](https://docs.ray.io/en/latest/rllib.html#), an open-source library for reinforcement learning(RL) and ease to execute efficient and scalable RL experiments due to integrated with Ray.

## Installation

### Install Anaconda

It is highly recommended to install [Anaconda](https://www.anaconda.com/products/individual) that is convenient to set up a specific environment for Flow and its dependencies.

### Install FLOW

Download this project, which covers the whole framework of [Flow](https://github.com/flow-project/flow) and my model implementation based on Flow.

```shell
git clone git@github.com:Guojyjy/AVARS.git
```

#### Create a conda environment

First, `cd ~/AVARS/flow` to the directory, then enter the following commands and run the relevant scripts to create the project environment and activate it, install Flow and its dependencies:

```shell
conda env create -f environment.yml
conda activate flow
python setup.py develop # if the conda install fails, try the next command to install the requirements using pip
pip install -e . # install flow within the environment
```

If you have trouble installing FLOW, the official Flow documentation provides more installation details: [Local installation of Flow](https://flow.readthedocs.io/en/latest/flow_setup.html#).

In addition, because the definition of $SUMO_HOME within the installation process of SUMO would cause an error in the installation of Flow, please confirm the installation of Flow first.

### Install SUMO

It is highly recommended to use the official installation methods from [Downloads-SUMO documentation](https://sumo.dlr.de/docs/Downloads.php). 

The instructions covered in [Installing Flow and SUMO](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo) from Flow documentation is outdated.

```shell
# run the following commands to check the version/location information or load SUMO GUI
which sumo
sumo --version
sumo-gui
```

#### Troubleshooting

1. See output like `Warning: Cannot find local schema '../sumo/data/xsd/types_file.xsd', will try website lookup.`
   - Set `$SUMO_HOME` to `$../sumo `instead of `$../sumo/bin`
2. Error like `ModuleNotFoundError: No module named 'flow'`, `ImportError: No module named flow.subpackage`
   - `pip install -e . ` to  install flow within the environment, mentioned at **Install FLOW**
   - Issue on inconsistent version of python required in the environment
     - Linux version of SUMO contains python in `/sumo/bin/` which may cause the error.
     - `which python` to check the current used
     - `echo $PATH` to check the order of the directories in the path variable to look for python
     - Add `export PATH=/../anaconda3/env/flow/bin:$PATH` in the file `~/.bashrc`
     - Restart the terminal,  update the configuration through`source ~/.bashrc`

## Experiments

Enter the project in the specific environment:
```shell
cd ~/AVARS/flow
conda activate flow
```

### Scenario settings

Extracted from the open data in [maxime-gueriau
/
ITSC2020_CAV_impact](https://github.com/maxime-gueriau/ITSC2020_CAV_impact) to simulate the real-world traffic in Dublin city.

#### Road network

A subnet of Dublin city center road network around the River Liffey, covering approximately 1 square kilometer, is illustrated as the traffic environment in the system flow diagram above.

#### Traffic 

Traffic generation lasts 45 minutes with 1168 vehicles.

### Compared scenarios

#### AVARS

```shell
# locate in ~/AVARS/flow
python examples/train_ppo.py AVARS_UAV --num_steps 150
python examples/train_dqn.py AVARS_UAV_dqn --num_steps 150
```

- ***train_ppo.py*** and ***train_dqn.py*** include DRL algorithm assigned and parameter settings
  - *num_gpu*, *num_worker* specify the computation resource used in the DRL training
- ***AVARS_UAV*** and ***AVARS_UAV_dqn*** both correspond to the modules of  ***flow/examples/exp_configs/rl/multiagent***, including all setting of road network and DRL agent 
- *"**--num_steps**"* is the termination condition for DRL training, optional
- The SUMO files of Dublin scenario locate in ***AVARS/scenarios/UAV***
- System design in ***flow/flow/env/multiagent/sumo_template.py***
  - ***UAVEnvAVARS***
  - ***UAVEnvIntelliLight***

#### Original

```shell
cd ~/AVARS/SCATS
python Imple_SCATS.py --scen center10 --nogui --noscats --run_num 18
```

#### Congestion

```shell
cd ~/AVARS/SCATS
python Imple_SCATS.py --scen center10_closing --nogui --noscats --run_num 18
```

#### SCATS[1]

```shell
cd ~/AVARS/SCATS
python Imple_SCATS.py --scen center10_SCATS --nogui --run_num 18
```
- SCATS requires [Induction Loops Detectors (E1)](https://sumo.dlr.de/docs/Simulation/Output/Induction_Loops_Detectors_%28E1%29.html) to collect traffic volumes

#### IntelliLight[2]

```shell
# locate in ~/AVARS/flow
python examples/train_ppo.py IntelliLight_UAV_ppo --num_steps 150
python examples/train_dqn.py IntelliLight_UAV --num_steps 150
```

### Evaluation

- **_evaluation/outputFilesProcessing.py_** filters the output files in AVARS/output

  - delete outdated, incomplete, and initial (size < 4kB ) files

  - recover the format of xml files

- **_evaluation/getResults.py_** gets traffic statistic

  - Travel time
  - Fuel consumption
  - CO2 emissions
  
  ```python getResults.py --scen center10 center10_closing center10_SCATS ... ```
- **_evaluation/draw_reward_episode.py_** generates lineplot of average episode reward for AVARS-controlled TLC, based on _progress.csv_ in each training stored at _ray_results/_

- **_evaluation/draw_runningveh.py_** uses the output from sumo-gui when running _Imple_SCATS.py_ or _flow/flow/visualize/visualizer_rllib.py_
-----

### Citing

```

```

### References

[1] _P. Lowrie, “SCATS, sydney co-ordinated adaptive traffic system: A traffic responsive method of controlling urban traffic,” 1990._

[2] _H. Wei, G. Zheng, H. Yao, and Z. Li, “Intellilight: A reinforcement learning approach for intelligent traffic light control,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 2496–2505._


