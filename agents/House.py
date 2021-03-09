import numpy as np

from constants import *
from agents import Agent
import helics as h

from dwelling_model import Dwelling

HOUSE_DEFAULT_ARGS = {
    # Timing parameters
    'start_time': start_time,
    'time_res': freq_house,
    'duration': duration,
    'times': times,
    'initialization_time': timedelta(days=7),

    # Input and Output Files
    # 'input_path': doom_input_path,
    'output_path': house_results_path,
    'weather_file': epw_weather_file_name,

    # 'envelope_model_name': 'Env',
    'assume_equipment': True,
    'uncontrolled_equipment': ['Lighting', 'Exterior Lighting', 'Range', 'Dishwasher', 'Refrigerator',
                               'Clothes Washer', 'Clothes Dryer', 'MELs', 'EV Scheduled'],  # List of uncontrolled loads
    'save_results': True,
    'verbosity': 7,
    'ext_time_res': freq_hems,
}

equipment = {

}


def update_house_args(house_id):
    house_args = HOUSE_DEFAULT_ARGS

    # Load Master Spreadsheet and get house information
    house_row = master_df.loc[house_id]
    house_args.update(house_row)

    # TODO: update the scale factor and the profile file
    # equipment['PV Scheduled'] = {
    #     'equipment_schedule_file': pv_profile_file,
    #     'schedule_scale_factor': -1 / house_args['Num_unit_in_building'],
    #     'val_col': house_args['PV_Profile_ID']
    # }

    # TODO: update the EV 
    # EV
    # equipment_dict['EV'] = {
    #         'vehicle_type': 'PHEV',
    #         'charging_level': 'Level1',
    #         'mileage': 20
    #     }

    # Battery #TODO: update
    # if house_args['Battery_size_{} (kWh)'.format(der_penetration_pc)] > 0:
    #     equipment['Battery'] = {
    #         'capacity_kwh': house_args['Battery_size_{} (kWh)'.format(der_penetration_pc)],
    #         'capacity_kw': 3,
    #         'parameter_file': os.path.join(doom_input_path, 'battery', 'default_parameters.csv'),
    #         'enable_schedule': False
    #     }

    # Properties file
    prop_file_path = os.path.join(doom_input_path, 'building_models', building_model)
    rc_file = house_args['DOOM Input File']
    house_args['properties_file'] = os.path.join(prop_file_path, rc_file)

    # Schedule file
    sched_file = rc_file.replace('rc_model', 'schedule')
    house_args['schedule_file'] = os.path.join(prop_file_path, sched_file)

    # Player Files
    house_args['water_draw_file'] = house_args['Draw Profile']

    # Output path
    house_args['output_path'] = os.path.join(house_results_path, 'House_{}'.format(house_id)) 
    return house_args


class House(Agent):
    def __init__(self, house_id, **kwargs):
        self.house_id = house_id
        print('_house_id')
        print(self.house_id)

        self.hems_controls = None
        self.status = {}
        self.house = None
        kwargs['result_path'] = os.path.join(house_results_path, 'House_{}'.format(house_id)) 
        super().__init__('House_' + house_id, **kwargs)

    def initialize(self):
        house_args = update_house_args(self.house_id)
        self.print_log(house_args)
        self.print_log(equipment)
        self.house = Dwelling(self.house_id, equipment, **house_args)

    def setup_pub_sub(self):
        topic_to_feeder_load = "load_to_feeder"
        self.register_pub("power", topic_to_feeder_load, h.helics_data_type_complex, global_type=False)

        topic_from_feeder = "Feeder/House_{}/voltage".format(self.house_id)
        self.register_sub("Feeder", topic_from_feeder, var_type=h.helics_data_type_complex)

        if include_hems:
            topic_to_ctrlr_status = "house_status_to_ctrlr_{}".format(self.house_id)
            self.register_pub("status", topic_to_ctrlr_status, h.helics_data_type_string)

            topic_from_ctrlr = "ctrlr_controls_to_house_{}".format(self.house_id)
            self.register_sub("controls", topic_from_ctrlr)

    def setup_actions(self):
        # Note: order matters!
        if include_hems:
            self.add_action(self.get_hems_controls, 'Get HEMS Controls', freq_hems, offset_house_pull_controls)
        self.add_action(self.run_house, 'Run House', freq_house, offset_house_run)
        if include_hems:
            self.add_action(self.send_status_to_hems, 'Send House Status', freq_hems, offset_house_run)
        self.add_action(self.save_results, 'Save Results', freq_save_results, offset_save_results)

    def get_voltage_from_feeder(self):
        voltage = self.fetch_subscription("Feeder")
        return 1 if voltage is None else (voltage[0] ** 2 + voltage[1] ** 2) ** 0.5 / 240.0   # convert voltage to p.u. values

    def send_powers_to_feeder(self, power_to_feeder):
        self.publish_to_topic("power", power_to_feeder)

    def send_status_to_hems(self):
        self.publish_to_topic("status", self.status)

    def get_hems_controls(self):
        self.hems_controls = self.fetch_subscription("controls")

    def run_house(self):
        voltage = self.get_voltage_from_feeder()

        # run simulator
        if include_hems and self.hems_controls and voltage:
            to_ext_control, to_feeder = self.house.update(voltage=voltage, from_ext_control=self.hems_controls)
        else:
            self.print_log("Did not receive house controls: {} and voltage {}".format(self.hems_controls, voltage))
            if voltage == 'None' or voltage == None: 
                to_ext_control, to_feeder = self.house.update(voltage=1)
            else:
                to_ext_control, to_feeder = self.house.update(voltage=voltage)
    
        self.send_powers_to_feeder(complex(to_feeder['P Total'] * 1000, to_feeder['Q Total'] * 1000)) # convert kW to W, kVAr to VAr
        self.status = {}
        for k, v in to_ext_control.items():
            if isinstance(v, np.ndarray):
                self.status[k] = v[0]
            else:
                self.status[k] = v
        self.print_log(self.status)
        
    def save_results(self):
        super().save_results()
        self.house.export_results()

    def finalize(self):
        self.house.finalize()
        # self.house.calculate_metrics()

        super().finalize()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        house_id = str(sys.argv[1])
        addr = str(sys.argv[2])
        agent = House(house_id, broker_addr=addr)
    elif len(sys.argv) == 2:
        house_id = str(sys.argv[1])
        agent = House(house_id, debug=True, run_helics=False)
    else:
        raise Exception('Must specify House ID')

    agent.simulate()
