import datetime as dt
import time
# import cProfile

from dwelling_model import Dwelling
from dwelling_model import CreateFigures
from bin.run_dwelling import dwelling_args

# Test script to run single equipment. Takes dwelling_args from run_dwelling

args = dwelling_args.copy()
args['duration'] = dt.timedelta(days=10)
args['assume_equipment'] = False
args['initialization_time'] = None

# input arguments for single equipment to test
equipment_dict = {
    'EV': {'vehicle_type': 'PHEV',
           'charging_level': 'Level1',
           'mileage': 20},
}

# optional dictionary to update the dwelling schedule
add_to_schedule = {
    # 'Indoor': 20,  # static indoor temp
}

if __name__ == '__main__':
    # Initialize dwelling, extract equipment
    dwelling = Dwelling('Test Equipment', equipment_dict, **args)

    equipment = dwelling.equipment[0]

    # Update schedule
    schedule = dwelling.schedule
    for key, val in add_to_schedule.items():
        schedule[key] = val

    # Simulate equipment
    t0 = time.time()
    df = equipment.simulate(schedule)
    # cProfile.run('df = equipment.simulate(schedule)', sort='cumulative')
    t1 = time.time()
    print('time to simulate: {}'.format(t1 - t0))

    fig = CreateFigures.plot_daily(df, equipment.name + ' Electric Power (kW)', plot_singles=False, plot_max=False,
                                   plot_min=False)
    fig.show()

    # plot all powers and temperatures
    power_cols = [col for col in df.columns if '(kW)' in col]
    if power_cols:
        df_powers = df.loc[:, power_cols].rename(columns={col: col[:-5] for col in power_cols})
        fig = CreateFigures.plot_time_series(df_powers, 'Power (kW)', legend=True)
        fig.show()

    temp_cols = [col for col in df.columns if '(C)' in col]
    if temp_cols:
        df_temps = df.loc[:, temp_cols].rename(columns={col: col[:-4] for col in temp_cols})
        fig = CreateFigures.plot_time_series(df_temps, 'Temperature (C)', legend=True)
        fig.show()
