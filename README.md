# Co-simulation for FAST-DERMS project

This setup is for co-simulation of Dwelling Object-Oriented Model (DOOM) and Gridlab-D using HELICS to demonstrate the 

The DOOM model generates load profiles for up to 40 selected `triplex_load` objects in the feeder model in Gridlab-D. The active and reactive power set points from DOOM will be sent to these loads in the model during the co-simulation process before solving power flow each time step.

The results are saved in `outputs folder` and log files are saved in `bin` folder.

### Installation

Build the docker image with

```shell
docker build -t fastdermscosim:latest .
```

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

To run the cosimulation, first start a docker container from the `fastdermscosim:latest` image

```shell
docker run -it fastdermscosim:latest
```

Inside the container,

 - activate the conda environment
 - create a config.json file and a gld_helics_config.json file
 - run the helics-runner script

```sh
conda activate fastderms
python bin/make_config_file.py localhost
helics run --path /bin/config.json --broker-loglevel=2
```

