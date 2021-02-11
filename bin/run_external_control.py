import datetime as dt

from dwelling_model import Dwelling
from bin.run_dwelling import dwelling_args, equipment

# Test script to run single Dwelling with constant external control signal

dwelling_args.update({
    'ext_time_res': dt.timedelta(minutes=15),
})

equipment.update({
    # 'PV': {
    #     'capacity': 5,
    #     # 'tilt': 20,
    #     # 'azimuth': 180,
    # },
    'Battery': {
        'capacity_kwh': 6,
        'capacity_kw': 3,
    },
})

from_ext_control = {
    'Sol_status_CVXPY__Dimensionless': 1,
    'HVAC Heating': {'Heating Setpoint': 19},  # , 'ER Duty Cycle': 0.1},
    'HVAC Cooling': {'Cooling Duty Cycle': 0.0},
    'Heat Pump Water Heater': {'HP Duty Cycle': 0.0, 'ER Duty Cycle': 0.0},
    'PV': {'P Setpoint': -1.1, 'Q Setpoint': 0.5},
    'Battery': {'P Setpoint': -1.0},
    'Load Fractions': {
        'Air Source Heat Pump': 1,
        'Heat Pump Water Heater': 0,
        'Electric Resistance Water Heater': 0,
        'Scheduled EV': 0,
        'Lighting': 0.2,
        'Exterior Lighting': 0.0,
        'Range': 0.0,
        'Dishwasher': 0.0,
        'Refrigerator': 1.0,
        'Clothes Washer': 0.0,
        'Clothes Dryer': 0.0,
        'MELs': 0.2,
    }
}

if __name__ == '__main__':
    # Initialization
    dwelling = Dwelling('Test House with Controller', equipment, **dwelling_args)

    # Simulation
    schedule_data = dwelling.schedule.to_dict('records')
    for current_schedule in schedule_data:
        dwelling.update(current_schedule, from_ext_control=from_ext_control)

    if dwelling.save_results:
        dwelling.export_results()
        print('Dwelling {} Complete, results saved to {}'.format(dwelling.name, dwelling.results_file))
