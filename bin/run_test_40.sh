#!/bin/bash

#SBATCH --account=corred
#SBATCH --time=3:00:00
#SBATCH --job-name=test40
#SBATCH --nodes=2
#SBATCH --mail-user=pmunanka@nrel.gov
#SBATCH --mail-type=FAIL
#SBATCH --output=test40.%j.out

source bin/scenario_max_flexibility_0.sh
    
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

python bin/make_config_file.py eagle
echo "Simulation Complete"

cd /home/pmunanka/proj/fastderms/Co-Simulation/outputs
zip -r test40.zip $SCENARIO_NAME
echo "All done. Checking results:"