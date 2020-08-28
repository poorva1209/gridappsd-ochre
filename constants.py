import os
import pandas as pd
from datetime import datetime, timedelta

# Scenario Name (used for finding Master Spreadsheet)
scenario_name = os.environ['SCENARIO_NAME'] if 'SCENARIO_NAME' in os.environ else 'test30'
no_of_homes = int(os.environ['NO_OF_HOMES']) if 'NO_OF_HOMES' in os.environ else 10
der_penetration_pc = os.environ['DER_PENETRATION'] if 'DER_PENETRATION' in os.environ else 'BAU'  # 'BAU', '50p', '100p'
building_model = os.environ['BUILDING_MODEL'] if 'BUILDING_MODEL' in os.environ else 'sd_ca'
debug = True

# Simulation interval
start_date = int(os.environ['START_DATE']) if 'START_DATE' in os.environ else 1
days = int(os.environ['NO_OF_DAYS']) if 'NO_OF_DAYS' in os.environ else 1
month = int(os.environ['MONTH']) if 'MONTH' in os.environ else 1
year = 2018
start_time = datetime(year, month, start_date, 0, 0)  # (Year, Month, Day, Hour, Min)
duration = timedelta(days=days)
time_step = timedelta(seconds=10)
end_time = start_time + duration
times = pd.date_range(start=start_time, end=end_time, freq=time_step)[:-1]

# Agents to run
include_house = os.environ['HOUSE'] == 'True' if 'HOUSE' in os.environ else True
include_feeder = os.environ['FEEDER'] == 'True' if 'FEEDER' in os.environ else True
include_hems = os.environ['HEMS'] == 'True' if 'HEMS' in os.environ else False

# Frequency of Updates
freq_house = timedelta(minutes=1)
freq_hems = timedelta(minutes=15)
freq_feeder = timedelta(minutes=1)
freq_save_results = timedelta(hours=1)

# Foresee variables 
hems_horizon = timedelta(hours = 8)

# Time offsets for communication order
offset_house_run = timedelta(seconds=0)
offset_feeder_run = timedelta(seconds=10)
offset_hems_run = timedelta(seconds=20)
offset_hems_to_house = timedelta(seconds=30)
offset_house_to_hems = timedelta(seconds=40)
offset_save_results = timedelta(0)

# Input/Output file paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
input_path = os.path.join(base_path, "inputs")
output_path = os.path.join(base_path, "outputs", scenario_name)
feeder_input_path = os.path.join(input_path, "opendss")
doom_input_path = os.path.join(input_path, "house")
foresee_input_path = os.path.join(input_path, "foresee")

# Input file locations
# TODO: update
if scenario_name == 'test':
    master_dss_file = os.path.join(feeder_input_path, "Secondary_p2udt2338_p2udt2338lv", "Master.dss")
elif scenario_name == 'test30':
    # master_dss_file = os.path.join(feeder_input_path, "test_40_houses", "Master.dss")
    master_dss_file = os.path.join(feeder_input_path, "IEEE123Bus", "Master.dss")
else:
    # TODO: Update once we have original feeder
    master_dss_file = os.path.join(feeder_input_path, "test_40_houses", "Master.dss")
print("MASTER DSS FILE:", master_dss_file)
epw_weather_file_name = os.path.join(input_path, 'weather', 'sd_ca_nsrdb_2018.csv')

# Output file locations
house_results_path = os.path.join(output_path, 'Dwelling Model')
hems_results_path = os.path.join(output_path, 'Foresee')
feeder_results_path = os.path.join(output_path, 'Feeder')

# processing master spreadsheet
# UPDATED THE NAME OF MS once we have final version
ms_file = os.path.join(input_path, "MS", "Main_spreadsheet_test40.xlsx")
master_df = pd.read_excel(ms_file, index_col='House_ID')[:no_of_homes]
house_ids = master_df.index.to_list()
feeder_loads = dict(zip(house_ids, master_df['Load_name']))
print(house_ids)
