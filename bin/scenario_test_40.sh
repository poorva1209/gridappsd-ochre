module purge
module load conda
source activate fastderms
export BASE_DIRECTORY=/home/pmunanka/proj/fastderms/Co-Simulation
export PYTHONPATH=$PYTHONPATH:$BASE_DIRECTORY
export SCENARIO_NAME="test40"
export NO_OF_HOMES="40"
export DER_PENETRATION='BAU'
export NO_OF_DAYS="365"
export START_DATE="1"
export MONTH="1"
export HOUSE="True"
export FEEDER="True"
export BUILDING_MODEL='sd_ca'