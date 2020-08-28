# Co-simulation for FAST-DERMS project

This setup is for co-simulation of Dwelling Object-Oriented Model (DOOM) and IEEE 123-Bus test system using HELICS to demonstrate the 

Includes agents for:
 - Feeder model in OpenDSS
 - House model using DOOM
 
The DOOM model generates load profiles for 10 selected loads in the IEEE test system modeled in OpenDSS. The active and reactive power set points from DOOM will be sent to these loads in the model during the co-simulation process before solving power flow each time step.

The results are saved in Co-simulation\outputs folder and log files are saved in Co-simulation\bin folder.

### Installation

We recommend using a `conda` environment.

If you do not have
[Anaconda](https://www.anaconda.com/distribution/) or
[miniconda](https://docs.conda.io/en/latest/miniconda.html),
download one of them. 

Windows: 
> Once it is downloaded, you will get an application called Anaconda Prompt.
> Next step is to create a conda environment , open the Anaconda Prompt and run:

Mac: 
> Once conda downloaded, you can use terminal to create to run the following commands

```
cd <path to Co-simulation folder>\Co-Simulation
conda env create
```

This will install all necessary Python packages in the conda environment as defined in environment.yml file.
> The environment.yml file might have dependencies with the repositories hosted in internal NREL github. We will update and test that shortly with github.com hosted repos. 

### Input Files

All scenario parameters can be adjusted from the `constants.py` file.
In this file, you can adjust the:
 - Simulation configuration
    - Number of homes 
    - Feeder model 
 - Simulation timing (start time, duration, etc.)
 - Input file location
 - Output file location
 - Many other high-level parameters

### Usage

To run the cosimulation:
 - open Anaconda Prompt
 - activate the conda environment
 - create a config.json file
 - run the helics-runner script

```
conda activate fastderms
cd <path to Co-simulation folder>\Co-Simulation
python bin/make_config_file.py localhost
helics run --path /bin/config.json --broker-loglevel=6
```

For mac users, we can define the bash scripts to define the environment variables (e.g. eagle.sh). The 'constants.py' file takes the environment variable if defined. 

```
cd <path to Co-simulation folder>\Co-Simulation
source eagle.sh
python bin/make_config_file.py localhost
helics run --path /bin/config.json --broker-loglevel=6
```

Note: Currently Windows users may experience a minor issue that the DOOM power set points for only 1 out of 10 loads would be passed to the OpenDSS model.