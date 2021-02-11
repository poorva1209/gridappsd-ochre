# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:24:32 2018

@author: mblonsky, kmckenna, jmaguire
"""

import os
import pandas as pd
import hashlib

from dwelling_model import FileIO, default_output_path, __version__
from dwelling_model.Models import *
from dwelling_model.Equipment import *

PROPERTIES_LOAD_NAMES = {
    'Lighting': 'lights',
    'Exterior Lighting': 'exterior lights',
    'Range': 'range',
    'Dishwasher': 'dishwasher',
    'Refrigerator': 'refrigerator',
    'Clothes Washer': 'clothes washer',
    'Clothes Dryer': 'clothes dryer',
    'MELs': 'misc electric',
}


class Dwelling:
    """
    A Dwelling is a collection of Equipment and an Envelope Model. All Equipment contribute to energy usage and most
    contribute to heat gains in the envelope. The Dwelling class also handles all input and output files, and defines
    the timing of the simulation.
    """

    def __init__(self, name, equipment_dict=None, seed=1, verbosity=1, assume_equipment=False,
                 uncontrolled_equipment=None, output_path=default_output_path, **house_args):
        print('Dwelling Object Oriented Model (DOOM), Version {}'.format(__version__))
        self.name = name

        # Time parameters
        self.start_time = house_args['start_time']
        self.time_res = house_args['time_res']
        self.end_time = self.start_time + house_args['duration']
        house_args['end_time'] = self.end_time
        self.current_time = self.start_time
        self.ext_time_res = house_args.get('ext_time_res')

        # Results parameters
        self.verbosity = verbosity
        self.save_results = house_args.pop('save_results', False)
        self.results = []
        if self.save_results:
            self.export_res = house_args.pop('export_res', None)
            os.makedirs(output_path, exist_ok=True)
            self.results_file = os.path.join(output_path, name + '.csv')
            if os.path.exists(self.results_file):
                print('Removing previous results for dwelling:', self.name)
                os.remove(self.results_file)
            self.metrics_file = os.path.join(output_path, name + '_metrics.csv')
            if os.path.exists(self.metrics_file):
                os.remove(self.metrics_file)
        else:
            self.export_res = None
            self.results_file = None
            self.metrics_file = None
        if self.save_results and self.verbosity >= 7:
            schedule_output_file = os.path.join(output_path, name + '_schedule.csv')
            if os.path.exists(schedule_output_file):
                os.remove(schedule_output_file)
            self.hourly_output_file = os.path.join(output_path, name + '_hourly.csv')
            if os.path.exists(self.hourly_output_file):
                os.remove(self.hourly_output_file)
        else:
            schedule_output_file = None
            self.hourly_output_file = None

        # Set random seed based on dwelling name, note that default uses fixed seed, i.e. no randomness
        if seed is None:
            seed = int(hashlib.md5(self.name.encode()).hexdigest(), 16) % 2 ** 32
        np.random.seed(seed)

        # Load properties file, add properties to house_args
        self.properties = FileIO.import_properties(**house_args)
        house_args = {**self.properties, **house_args}

        # Prepare equipment list
        if equipment_dict is None:
            equipment_dict = {}
        if 'Air Source Heat Pump' in equipment_dict:
            # remove ASHP and split into heater and cooler, move to front of dict
            hvac_dict = equipment_dict.pop('Air Source Heat Pump')
            equipment_dict = {'ASHP Heater': hvac_dict, 'ASHP Cooler': hvac_dict, **equipment_dict}
        if assume_equipment:
            equipment_dict = self.add_equipment_from_properties(equipment_dict)

        # Load all time series files: schedule, weather, solar gains, water draw, PV profile
        if any([name in EQUIPMENT_BY_END_USE['PV'] and 'equipment_schedule_file' not in data for name, data in
                equipment_dict.items()]):
            house_args['create_sam_file'] = True
        if any([name in EQUIPMENT_BY_END_USE['Water Heating'] for name in equipment_dict]) and \
                house_args.get('water_draw_file') is None:
            # choose random water draw file (0-9) based on number of occupants
            #TODO: Jeff removing randomness to better match EPlus/ResStock (for now)
            occupant_to_bedroom = [1, 2, 4, 5]
            bedrooms = np.searchsorted(occupant_to_bedroom, house_args.get('number of occupants', 0)) + 1
            house_args['water_draw_file'] = 'DHW_{}bed_unit{}_1min.csv'.format(bedrooms, 0)
        full_schedule = FileIO.import_all_time_series(schedule_output_file, **house_args)
        current_schedule = full_schedule.loc[self.start_time].to_dict()  # initial schedule
        house_args['initial_schedule'] = current_schedule
        house_args['full_schedule'] = full_schedule

        # keep only relevant columns for schedule
        keep_cols = ['Occupancy', 'heating_setpoint', 'cooling_setpoint', 'ventilation_rate',
                     'ambient_dry_bulb', 'ground_temperature', 'ambient_humidity', 'ambient_pressure', 'wind_speed',
                     'H_RF1', 'H_EW1', 'H_WD1', 'H_GW1',
                     'water_draw', 'tempered_water_draw', 'shower_draw', 'mains_temperature']
        self.schedule = full_schedule.loc[:, [col for col in keep_cols if col in full_schedule.columns]]
        self.schedule_iterable = self.schedule.itertuples()

        # Envelope model inputs
        self.envelope = Envelope(**house_args)
        house_args['envelope_model'] = self.envelope

        # Load ZIP file
        zip_model = FileIO.import_zip_model(**house_args)
        house_args['zip_model'] = zip_model
        self.voltage = 1

        # Power inputs
        self.total_p_kw = 0
        self.total_q_kvar = 0
        self.total_gas_therms_per_hour = 0

        # Create all equipment
        self.equipment = []
        for equipment_name, equipment_args in equipment_dict.items():
            if equipment_name not in EQUIPMENT_BY_NAME:
                print('WARNING: Skipping "{}", no equipment defined with that name.'.format(equipment_name))
            equipment_args = {**house_args, **equipment_args}
            equipment = EQUIPMENT_BY_NAME[equipment_name](name=equipment_name, **equipment_args)
            self.equipment.append(equipment)

        # sort equipment by end use
        self.equipment_by_end_use = {
            end_use: [e for e in self.equipment if e.end_use == end_use] for end_use in EQUIPMENT_BY_END_USE.keys()
        }
        # check if there is more than 1 equipment per end use. For now, raise error for HVAC/WH, else print a warning
        for end_use, equipment in self.equipment_by_end_use.items():
            if len(equipment) > 1:
                if end_use in ['HVAC Heating', 'HVAC Cooling', 'Water Heating']:
                    raise Exception('More than 1 equipment defined for {}: {}'.format(end_use, equipment))
                elif end_use != 'Other':
                    print('WARNING: More than 1 equipment defined for {}: {}'.format(end_use, equipment))

        # force Ideal HVAC equipment to go last - so all heat from other equipment is known during update
        for e in self.equipment_by_end_use['HVAC Heating'] + self.equipment_by_end_use['HVAC Cooling']:
            if isinstance(e, IdealHVAC):
                self.equipment.pop(self.equipment.index(e))
                self.equipment += [e]
        # force battery to go last - so it can run self-consumption controller
        for e in self.equipment_by_end_use['Battery']:
            self.equipment.pop(self.equipment.index(e))
            self.equipment += [e]

        # get list of uncontrolled equipment
        self.uncontrolled_p_kw = 0
        if uncontrolled_equipment is None:
            self.uncontrolled_equipment = []
        else:
            self.uncontrolled_equipment = [e for e in self.equipment if e.name in uncontrolled_equipment]

        # Run initialization to get realistic initial state
        initialization_time = house_args.get('initialization_time')
        if initialization_time is not None:
            self.initialize(initialization_time)

            # if initialization time > duration, reset the schedule to just the duration
            if initialization_time > house_args['duration']:
                self.schedule = self.schedule.loc[self.schedule.index < self.end_time]

        print('Dwelling {} Initialized'.format(self.name))

    def add_equipment_from_properties(self, equipment_dict):
        # add or update equipment from properties. Values in equipment_dict have higher priority
        # Note: PV, Battery, and EV are not taken from properties file

        # HVAC heating
        heating_names = {('electric', 'Furnace'): ('Electric Furnace', {}),
                         ('electric', 'Baseboard'): ('Electric Baseboard', {}),
                         ('electric', 'Boiler'): ('Electric Boiler', {}),
                         ('electric', 'ASHP'): ('ASHP Heater', {}),
                         ('electric', 'Ideal ASHP'): ('Ideal ASHP Heater', {}),
                         ('electric', 'Ideal Heater'): ('Ideal Heater', {}),
                         ('gas', 'Furnace'): ('Gas Furnace', {}),
                         ('gas', 'Boiler'): ('Gas Boiler', {}),
                         }
        heating_type = (self.properties.get('heating fuel', 'None'),
                        self.properties.get('heating equipment type', 'None'))
        if any([heater in equipment_dict for heater in EQUIPMENT_BY_END_USE['HVAC Heating']]):
            # don't add the default heater if one exists already
            pass
        elif heating_type == ('None', 'None'):
            # don't add a heater if not specified in the properties file
            pass
        elif heating_type in heating_names:
            heater_name, heater_dict = heating_names[heating_type]
            equipment_dict[heater_name] = heater_dict
        else:
            raise Exception('Unknown heating equipment type: {}'.format(heating_type))

        # HVAC cooling
        cooling_names = {'Central AC': ('Air Conditioner', {}),
                         'Room AC': ('Room AC', {}),
                         'ASHP': ('ASHP Cooler', {}),
                         'Ideal Room AC': ('Ideal Room AC', {}),
                         'Ideal ASHP': ('Ideal ASHP Cooler', {}),
                         'Ideal Cooler': ('Ideal Cooler', {}),
                         }
        cooling_type = self.properties.get('cooling equipment type', 'None')
        if any([cooler in equipment_dict for cooler in EQUIPMENT_BY_END_USE['HVAC Cooling']]):
            # don't add the default cooler if one exists already
            pass
        elif cooling_type == 'None':
            # don't add a cooler if not specified in the properties file
            pass
        elif cooling_type in cooling_names:
            cooler_name, cooler_dict = cooling_names[cooling_type]
            equipment_dict[cooler_name] = cooler_dict
        else:
            raise Exception('Unknown cooling equipment type: {}'.format(cooling_type))

        # Water Heater
        wh_names = {('Electric', 'Tank'): 'Electric Resistance Water Heater',
                    ('Electric', 'Tankless'): 'Tankless Water Heater',
                    ('Electric', 'Heatpump'): 'Heat Pump Water Heater',
                    ('Electric', 'Ideal'): 'Modulating Water Heater',
                    ('Gas', 'Tank'): 'Gas Water Heater',
                    ('Gas', 'Tankless'): 'Gas Tankless Water Heater',
                    }
        wh_type = self.properties.get('water heater fuel', 'None'), self.properties.get('water heater type', 'None')
        if any([wh in equipment_dict for wh in EQUIPMENT_BY_END_USE['Water Heating']]):
            # don't add the default water heater if one exists already
            pass
        elif wh_type == ('None', 'None'):
            # don't add a water heater if not specified in the properties file
            pass
        elif wh_type in wh_names:
            wh_name = wh_names[wh_type]
            equipment_dict[wh_name] = {}
        else:
            raise Exception('Unknown water heating equipment type: {}'.format(wh_type))

        # add Scheduled Loads
        for equipment, properties_name in PROPERTIES_LOAD_NAMES.items():
            if properties_name + ' convective gainfrac' in self.properties and equipment not in equipment_dict:
                equipment_dict[equipment] = {'properties_name': properties_name}
                if 'Exterior' in equipment:
                    equipment_dict[equipment]['air_node'] = None

        return equipment_dict

    def update(self, current_schedule=None, voltage=1, from_ext_control=None):
        if np.isnan(voltage) or voltage <= 0:
            raise Exception('Error reading voltage for house {}: {}'.format(self.name, voltage))
        self.voltage = voltage

        if from_ext_control is None:
            from_ext = {}
        else:
            # Parse data from external controller
            from_ext = from_ext_control.copy()
            for key, val in from_ext_control.items():
                if key in self.equipment_by_end_use:
                    # copy data to all equipment with the end use
                    for e in self.equipment_by_end_use[key]:
                        if e.name not in from_ext_control:
                            from_ext[e.name] = {}
                        from_ext[e.name].update(from_ext_control[key])
            # TODO: remove this option
            if 'Load Fractions' in from_ext_control:
                for name, val in from_ext_control['Load Fractions'].items():
                    if name in from_ext:
                        from_ext[name]['Load Fraction'] = val
                    else:
                        from_ext[name] = {'Load Fraction': val}

        # Update schedule with current time and envelope states
        if current_schedule is None:
            current_schedule = next(self.schedule_iterable)._asdict()
        current_schedule.update(self.envelope.get_main_states())
        current_schedule['Indoor Humidity Ratio'] = self.envelope.humidity.indoor_w

        # Initialize outputs
        to_envelope = dict(zip(self.envelope.input_names, self.envelope.default_u))
        liv_sensible = 0
        liv_latent = 0

        # Update equipment
        self.total_p_kw = 0
        self.total_q_kvar = 0
        self.total_gas_therms_per_hour = 0
        self.uncontrolled_p_kw = 0
        for equipment in self.equipment:
            equip_from_ext = from_ext.get(equipment.name, {})
            if isinstance(equipment, IdealHVAC):
                to_env = to_envelope.copy()
                to_env['H_LIV'] += liv_sensible
                to_env['H_LIV_latent'] = liv_latent
                current_schedule.update({'to_envelope': to_env})
            elif isinstance(equipment, Battery):
                pv_power = sum([e.electric_kw for e in self.equipment_by_end_use['PV']])
                current_schedule.update({'net_power': self.total_p_kw,
                                         'pv_power': pv_power})

            if equipment.current_time <= self.current_time:
                equipment.update(self.voltage, current_schedule, equip_from_ext)

            # Add equipment heat gains to envelope - only liv node for now
            if equipment.air_node == 'LIV':
                # TODO: move back to 1e5
                if not isinstance(equipment.sensible_gain, (int, float)) or abs(equipment.sensible_gain) > 1e6:
                    raise EquipmentException(
                        'Bad heat output for {}: {} W'.format(equipment.name, equipment.sensible_gain))
                liv_sensible += equipment.sensible_gain
                liv_latent += equipment.latent_gain
            elif equipment.air_node is not None and equipment.sensible_gain != 0:
                print('WARNING: Ignoring heat gains from {} into {} node'.format(equipment.name, equipment.air_node))

            # update total powers, including uncontrolled electric loads
            self.total_p_kw += equipment.electric_kw
            self.total_q_kvar += equipment.reactive_kvar
            self.total_gas_therms_per_hour += equipment.gas_therms_per_hour
            if equipment in self.uncontrolled_equipment:
                self.uncontrolled_p_kw += equipment.electric_kw

        # # Update water model (currently done inside the Water Heater equipment)
        # water_to_ext_control = self.water_model.update(**current_schedule)
        # to_ext_control.update(water_to_ext_control)

        # Update envelope model
        to_envelope['H_LIV'] += liv_sensible
        to_envelope['H_LIV_latent'] = liv_latent
        self.envelope.update(to_envelope, current_schedule)

        # Update outputs to external controller
        if self.ext_time_res is not None and not (self.current_time - self.start_time) % self.ext_time_res:
            to_ext_control = {
                'House': {'P Total': self.total_p_kw,
                          'Q Total': self.total_q_kvar,
                          'P Uncontrolled': self.uncontrolled_p_kw},
                'Envelope': self.envelope.generate_results(self.verbosity, to_ext=True),
            }
            for equipment in self.equipment:
                to_ext_control.update(equipment.generate_results(self.verbosity, to_ext=True))
        else:
            to_ext_control = {}

        # update outputs to grid
        to_grid = {'P Total': self.total_p_kw,
                   'Q Total': self.total_q_kvar}

        # Update Results
        if self.save_results:
            self.results.append(self.compile_results())

        # Update time
        self.current_time += self.time_res

        return to_ext_control, to_grid

    def compile_results(self):
        # Results columns are in this order (minimum verbosity level):
        # 1. Total house power in kW: P, Q, Gas (0)
        # 2. Total house energy in kWh: P, Q, Gas (1)
        # 3. Electric and/or gas power by end use (2)
        # 4. House voltage and reactive power by end use (5)
        # 5. Envelope results:
        #    - Air temperatures from main zones, includes wet bulb (1)
        #    - Humidity, infiltration, and convection results (4)
        #    - Detailed model results (8)
        # 6. Specific equipment results, including:
        #    - HVAC heat delivered (3)
        #    - Water tank main results (3)
        #    - Battery and EV SOC (3)
        #    - All other equipment results (6)
        # 8. Water tank model detailed results (9)

        hours_per_step = self.time_res / dt.timedelta(hours=1)
        out = {'Time': self.current_time,
               'Total Electric Power (kW)': self.total_p_kw,
               'Total Reactive Power (kVAR)': self.total_q_kvar,
               'Total Gas Power (therms/hour)': self.total_gas_therms_per_hour
               }

        if self.verbosity >= 1:
            out.update({
                'Total Electric Energy (kWh)': self.total_p_kw * hours_per_step,
                'Total Reactive Energy (kVARh)': self.total_q_kvar * hours_per_step,
                'Total Gas Consumption (therms)': self.total_gas_therms_per_hour * hours_per_step,
            })

        if self.verbosity >= 2:
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_electric for e in equipment]):
                    out[end_use + ' Electric Power (kW)'] = sum([e.electric_kw for e in equipment])
            if self.uncontrolled_equipment:
                out['Uncontrollable Electric Power (kW)'] = self.uncontrolled_p_kw
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_gas for e in equipment]):
                    out[end_use + ' Gas Power (therms/hour)'] = sum([e.gas_therms_per_hour for e in equipment])
        if self.verbosity >= 5:
            out['Voltage (-)'] = self.voltage
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_electric for e in equipment]):
                    out[end_use + ' Reactive Power (kVAR)'] = sum([e.reactive_kvar for e in equipment])

        out.update(self.envelope.generate_results(self.verbosity))

        for equipment in self.equipment_by_end_use.values():
            for e in equipment:
                out.update(e.generate_results(self.verbosity))

        return out

    def export_results(self):
        df = pd.DataFrame(self.results)
        if os.path.exists(self.results_file):
            df.to_csv(self.results_file, index=False, header=False, mode='a')
        else:
            df.to_csv(self.results_file, index=False)

        self.results = []

    def calculate_metrics(self, results=None):
        if results is None:
            results = pd.read_csv(self.results_file, index_col='Time', parse_dates=True)

        hr_per_step = self.time_res / dt.timedelta(hours=1)
        missing = pd.Series([np.nan] * len(results), index=results.index)
        p = results.get('Total Electric Power (kW)', missing)
        q = results.get('Total Reactive Power (kVAR)', missing)
        g = results.get('Total Gas Power (therms/hour)', missing)

        # Main metrics
        metrics = {
            'Total Electric Energy (kWh)': p.sum(skipna=False) * hr_per_step,
            'Average Electric Power (kW)': p.mean(),
            'Peak Electric Power (kW)': p.max(),
            'Peak Electric Power - 15 min avg (kW)': p.resample('15min').mean().max(),
            'Peak Electric Power - 30 min avg (kW)': p.resample('30min').mean().max(),
            'Peak Electric Power - 1 hour avg (kW)': p.resample('1H').mean().max(),
            'Total Reactive Energy (kVARh)': q.sum(skipna=False) * hr_per_step,
            'Total Gas Energy (therms)': g.sum(skipna=False) * hr_per_step,
        }

        # Equipment power metrics
        power_names = [('Electric Power (kW)', 'Electric Energy (kWh)'),
                       ('Gas Power (therms/hour)', 'Gas Energy (therms)'),
                       ('Reactive Power (kVAR)', 'Reactive Energy (kVARh)')]
        for power_name, energy_name in power_names:
            for end_use in ALL_END_USES:
                power = results.get(' '.join([end_use, power_name]), missing)
                metrics[' '.join([end_use, energy_name])] = power.sum(skipna=False) * hr_per_step

        # Envelope metrics
        for node in MAIN_NODES.values():
            col = 'Temperature - {} (C)'.format(node)
            metrics['Average ' + col] = results.get(col, missing).mean()
        metrics['Std. Dev. Temperature - Indoor (C)'] = results.get('Temperature - Indoor (C)', missing).std()

        # HVAC metrics
        if 'Unmet HVAC Load (C)' in results:
            unmet_hvac = results.get('Unmet HVAC Load (C)', missing)
            metrics['Unmet Heating Load (C-hours)'] = -unmet_hvac.clip(upper=0).sum(skipna=False) * hr_per_step
            metrics['Unmet Cooling Load (C-hours)'] = unmet_hvac.clip(lower=0).sum(skipna=False) * hr_per_step
        for hvac_type in ['HVAC Heating', 'HVAC Cooling']:
            # Delivered heating/cooling
            delivered = results.get(hvac_type + ' Delivered (kW)', missing)
            delivered_total = delivered.sum(skipna=False)
            metrics['Total {} Delivered (kWh)'.format(hvac_type)] = delivered_total * hr_per_step

            # COP and capacity - weighted average only when device is on
            capacity = results.get(hvac_type + ' Capacity (tons)', missing)
            cop = results.get(hvac_type + ' COP (-)', missing)
            shr = results.get(hvac_type + ' SHR (-)', missing)
            metrics['Average {} Capacity (tons)'.format(hvac_type)] = capacity[capacity > 0].mean()
            if not np.isnan(delivered_total) and delivered_total != 0:
                metrics['Average {} COP (-)'.format(hvac_type)] = (cop * delivered).sum(skipna=False) / delivered_total
                metrics['Average {} SHR (-)'.format(hvac_type)] = (shr * delivered).sum(skipna=False) / delivered_total
            else:
                metrics['Average {} COP (-)'.format(hvac_type)] = np.nan
                metrics['Average {} SHR (-)'.format(hvac_type)] = np.nan

        # Water heater metrics
        heat = results.get('Water Heating Delivered (kW)', missing)
        metrics['Total Water Heating Delivered (kWh)'] = heat.sum(skipna=False) * hr_per_step
        # COP - weighted average only when device is on
        cop = results.get('Water Heating COP (-)', missing)
        on_times = cop > 0
        metrics['Average Water Heating Delivered (kW)'] = heat[on_times].mean()
        metrics['Average Water Heating COP (-)'] = (cop * heat).sum(skipna=False) / heat.sum(skipna=False)

        # FUTURE: Down with imperial units!
        metrics['Total Hot Water Delivered (gal)'] = Units.liter2pint(
            results.get('Hot Water Delivered (L/min)', missing).sum(skipna=False) * hr_per_step * 60) / 8
        metrics['Total Hot Water Delivered (kWh)'] = \
            results.get('Hot Water Delivered (kW)', missing).sum(skipna=False) * hr_per_step
        metrics['Total Hot Water Unmet Demand, Showers (kWh)'] = \
            results.get('Hot Water Unmet Demand, Showers (kW)', missing).sum(skipna=False) * hr_per_step

        # Equipment power, cycling, and timing metrics
        for e in self.equipment:
            if e.end_use == 'Other' or len(self.equipment_by_end_use[e.end_use]) > 1:
                # include total power metrics
                for power_name, energy_name in power_names:
                    power = results.get(' '.join([e.name, power_name]), missing)
                    metrics[' '.join([e.name, energy_name])] = power.sum(skipna=False) * hr_per_step
                name = e.name
            else:
                name = e.end_use

            if e.modes == ['On', 'Off']:
                if e.mode_cycles['On'] > 1:
                    metrics[name + ' Cycles'] = e.mode_cycles['On']
            else:
                cycles = {'{} "{}" Cycles'.format(name, mode): cycles for mode, cycles in e.mode_cycles.items()}
                cycles.pop('{} "Off" Cycles'.format(name))
                if sum(cycles.values()) > 1:
                    metrics.update(cycles)

        # FUTURE: add rates, emissions, other post processing
        # print('Loading rate file...')
        # rate_file = os.path.join(main_path, 'Inputs', 'Rates', 'Utility Rates.csv')
        # df_rates = Input_File_Functions.import_generic(rate_file, keep_cols=locations, annual_output=True, **default_args)
        # df_rates.index.name = 'Time'
        # df_rates = df_rates.reset_index().melt(id_vars='Time', var_name='Location', value_name='Rate')
        #
        # print('Calculating annual costs...')
        # df_all = df_all.reset_index().merge(df_rates, how='left', on=['Time', 'Location']).set_index('Time')
        # df_all['Cost'] = df_all['Rate'] * df_all['Total Electric Energy (kWh)']
        # annual_costs = df_all.groupby(['Location', 'Setpoint Difference'])['Cost'].sum()
        # print(annual_costs)
        # annual_costs.to_csv(os.path.join(main_path, 'Outputs', 'poster_results.csv'))
        # df_all.reset_index().to_feather(os.path.join(main_path, 'Outputs', 'poster_all_data.feather'))

        # Save metrics to file (as single row df)
        df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        if self.metrics_file is not None:
            df.to_csv(self.metrics_file, index=False)
            print('Post-processing metrics saved to {}'.format(self.metrics_file))
        return metrics

    def initialize(self, duration):
        # run for duration, then reset time, don't save results
        print('Running {} Initialization for {}'.format(self.name, duration))
        initialize_times = pd.date_range(self.start_time, self.start_time + duration, freq=self.time_res, closed='left')
        tmp = self.save_results
        self.save_results = False

        # schedule_data = self.schedule.loc[initialize_times].to_dict('records')
        # for current_schedule in schedule_data:
        init_schedule = self.schedule.loc[initialize_times]
        for current_schedule in init_schedule.itertuples():
            self.update(current_schedule=current_schedule._asdict())

        # Set time back to start
        self.current_time = self.start_time
        for e in self.equipment:
            e.reset_time()
        self.save_results = tmp

    def finalize(self):
        # save final results
        if self.save_results:
            self.export_results()
            print('{} Simulation Complete, results saved to {}'.format(self.name, self.results_file))

            # Post processing - calculate metrics and return data frames
            df = pd.read_csv(self.results_file, index_col='Time', parse_dates=True)
            metrics = self.calculate_metrics(df)

            # Convert to hourly data and save
            if self.hourly_output_file is not None:
                df_hourly = df.resample(dt.timedelta(hours=1)).mean()
                df_hourly.reset_index().to_csv(self.hourly_output_file, index=False)
                print('Hourly results saved to {}'.format(self.hourly_output_file))

            return df, metrics
        else:
            print('{} Simulation Complete'.format(self.name))
            return

    def simulate(self):
        print('Running {} Simulation for {}'.format(self.name, self.end_time - self.start_time))
        # schedule_data = self.schedule.to_dict('records')
        # for current_schedule in schedule_data:
        for current_schedule in self.schedule.itertuples():
            self.update(current_schedule=current_schedule._asdict())

            # Export results to file once per day
            if self.export_res is not None and not (self.current_time - self.start_time) % self.export_res:
                self.export_results()

        return self.finalize()
