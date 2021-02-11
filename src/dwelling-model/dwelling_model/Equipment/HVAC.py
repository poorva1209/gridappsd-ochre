# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:24:32 2018

@author: kmckenna
"""
import datetime as dt
import numpy as np

from dwelling_model.Equipment import Equipment, EquipmentException
from dwelling_model import Psychrometrics_HVAC, Units

PsyCalc = Psychrometrics_HVAC.Psychrometrics()

SPEED_TYPES = ['Single', 'Double', 'Variable']

cp_air = 1.005  # kJ/kg/K
rho_air = 1.2041  # kg/m^3


class HVAC(Equipment):
    """
    Base HVAC Equipment Class. Assumes static capacity. `hvac_type` must be specified in child classes
    """
    name = 'Generic HVAC'
    hvac_type = None  # Options are 'heat' and 'cool'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.hvac_type == 'heat':
            self.is_heater = True
        elif self.hvac_type == 'cool':
            self.is_heater = False
        else:
            raise EquipmentException('HVAC type for {} Equipment must be "heat" or "cool".'.format(self.name))

        # Capacity parameters
        self.capacity_rated = kwargs[self.hvac_type + 'ing capacity (W)']  # rated capacity in W
        self.capacity = self.capacity_rated
        self.delivered_heat = 0  # in W
        self.space_fraction = kwargs.get(self.hvac_type + 'ing conditioned space fraction', 1.0)

        # Efficiency/loss parameters
        self.eir_rated = kwargs[self.hvac_type + 'ing EIR']  # Energy Input Ratio
        if isinstance(self.eir_rated, str):
            # For now, just take first value of the list
            self.eir_rated = float(self.eir_rated.strip('][').split(', ')[0])
        self.eir = self.eir_rated
        self.duct_dse = kwargs[self.hvac_type + 'ing duct dse']  # Duct distribution system efficiency
        if self.is_heater:
            self.fan_power_rated = kwargs['heating fan power (W)']  # power output from fan, in W
        else:
            self.fan_power_rated = kwargs['cooling fan power']  # power output from fan, in W
        self.fan_power = 0

        # Humidity and SHR parameters, cooling only
        self.coil_input_air_wb = None  # Wet bulb temperature after increase from fan power
        if not self.is_heater:
            self.shr_rated = kwargs[self.hvac_type + 'ing SHR']
            self.volumetric_flow_rate = Units.cfm2m3_s(kwargs[self.hvac_type + 'ing airflow rate (cfm)'])
            rated_dry_bulb = Units.F2C(80)  # in degrees C
            rated_wet_bulb = Units.F2C(67)  # in degrees C
            rated_pressure = 101.3  # in kPa
            self.Ao = PsyCalc.CoilAoFactor_SI(rated_dry_bulb, rated_wet_bulb, rated_pressure, self.capacity / 1000,
                                              self.volumetric_flow_rate, self.shr_rated)  # Coil Ao factor
        else:
            self.shr_rated = 1
            self.volumetric_flow_rate = None
            self.Ao = None
        self.shr = self.shr_rated

        # Thermostat Control Parameters
        self.temp_setpoint = kwargs['initial_schedule']['{}ing_setpoint'.format(self.hvac_type)]
        self.temp_deadband = kwargs.get(self.hvac_type + 'ing deadband temperature (C)', 1)
        self.ext_ignore_thermostat = kwargs.get('ext_ignore_thermostat', False)
        self.temp_indoor_prev = self.temp_setpoint

        # Minimum On/Off Times
        on_time = kwargs.get(self.hvac_type + 'ing Minimum On Time', 0)
        off_time = kwargs.get(self.hvac_type + 'ing Minimum Off Time', 0)
        self.min_time_in_mode = {mode: dt.timedelta(minutes=on_time) for mode in self.modes}
        self.min_time_in_mode['Off'] = dt.timedelta(minutes=off_time)

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces HVAC off)
        # - Heating (Cooling) Setpoint: Updates heating (cooling) setpoint temperature from the dwelling schedule (in C)
        #   - Note: Setpoint must be provided every timestep or it will revert back to the dwelling schedule
        # - Heating (Cooling) Deadband: Updates heating (cooling) deadband temperature (in C)
        #   - Note: Deadband will not reset back to original value
        # - Heating (Cooling) Duty Cycle: Forces HVAC on for fraction of external time step (as fraction [0,1])
        #   - If 0 < Duty Cycle < 1, the equipment will cycle once every 2 external time steps
        #   - For ASHP: Can supply HP and ER duty cycles
        #   - Note: does not use clock on/off time

        # If load fraction = 0, force off
        load_fraction = ext_control_args.get('Load Fraction', 1)
        if load_fraction == 0:
            return 'Off'
        elif load_fraction != 1:
            raise EquipmentException("{} can't handle non-integer load fractions".format(self.name))

        ext_setpoint = ext_control_args.get('{}ing Setpoint'.format(self.hvac_type.capitalize()))
        if ext_setpoint is not None:
            schedule = schedule.copy()
            schedule['{}ing_setpoint'.format(self.hvac_type)] = ext_setpoint

        ext_db = ext_control_args.get('{}ing Deadband'.format(self.hvac_type.capitalize()))
        if ext_db is not None:
            self.temp_deadband = ext_db

        if any(['Duty Cycle' in key for key in ext_control_args]):
            return self.run_duty_cycle_control(schedule, ext_control_args)
        else:
            return self.update_internal_control(schedule)

    def update_internal_control(self, schedule):
        # Run thermostat controller
        new_mode = self.run_thermostat_control(schedule)

        # Override mode switch with minimum on/off times
        new_mode = self.update_clock_on_off_time(new_mode)

        return new_mode

    def run_duty_cycle_control(self, schedule, ext_control_args):
        duty_cycles = ext_control_args.get('{}ing Duty Cycle'.format(self.hvac_type.capitalize()))

        # Set mode based on duty cycle from external controller
        if isinstance(duty_cycles, (int, float)):
            duty_cycles = [duty_cycles]
        mode_priority = self.calculate_mode_priority(*duty_cycles)
        thermostat_mode = self.run_thermostat_control(schedule)

        if thermostat_mode in mode_priority and not self.ext_ignore_thermostat:
            return thermostat_mode
        else:
            return mode_priority[0]  # take highest priority mode (usually current mode)

    def run_thermostat_control(self, schedule):
        temp_indoor = schedule['Indoor']

        # Update setpoint from schedule file
        self.temp_setpoint = schedule['{}ing_setpoint'.format(self.hvac_type)]

        # On and off limits depend on heating vs. cooling
        mult = 1 if self.is_heater else -1
        temp_turn_on = self.temp_setpoint - mult * self.temp_deadband / 2
        temp_turn_off = self.temp_setpoint + mult * self.temp_deadband / 2

        # Determine mode
        if mult * (temp_indoor - temp_turn_on) < 0:
            return 'On'
        if mult * (temp_indoor - temp_turn_off) > 0:
            return 'Off'

    def update_clock_on_off_time(self, new_mode):
        # This logic ensures that the HVAC abides by minimum on and off time operating requirements
        prev_mode = self.mode
        if new_mode is not None and self.time_in_mode < self.min_time_in_mode[prev_mode]:
            # Force mode to remain as is
            new_mode = prev_mode

        return new_mode

    def update_capacity(self, schedule):
        # set to rated values when on, set to 0 when off
        self.capacity = self.capacity_rated

    def update_eir(self, schedule):
        # set to rated values when on, set to 0 when off
        self.eir = self.eir_rated

    def update_shr(self, schedule):
        if self.is_heater:
            self.shr = self.shr_rated
            return

        t_in_db = schedule['Indoor']
        # t_in_wb = schedule['Indoor Wet Bulb']
        w_in = schedule['Indoor Humidity Ratio']
        pres_ext = schedule['ambient_pressure']
        pres_int = pres_ext

        # increase coil input temp based on fan power - assumes fan power from previous time step
        t_in_db += self.fan_power / 1000 / self.volumetric_flow_rate / rho_air / cp_air

        # calculate new wet bulb temperature
        rh_in = np.clip(PsyCalc.R_fT_w_P_SI(t_in_db, w_in, pres_int), 0, 1)
        t_in_wb = PsyCalc.Twb_fT_R_P_SI(t_in_db, rh_in, pres_int)
        w_new = PsyCalc.w_fT_Twb_P_SI(t_in_db, t_in_wb, pres_int)
        assert abs(w_new - w_in) < 1e-3
        self.coil_input_air_wb = t_in_wb

        if rh_in == 0:
            self.shr = 1
        else:
            self.shr = PsyCalc.CalculateSHR_SI(t_in_db, t_in_wb, pres_int, self.capacity / 1000,
                                               self.volumetric_flow_rate, self.Ao)

    def calculate_power_and_heat(self, schedule):
        # Calculate delivered heat to envelope model
        if 'On' in self.mode:
            self.update_shr(schedule)
            self.update_capacity(schedule)
            self.update_eir(schedule)
        else:
            # if 'Off', set capacity and EIR to 0 (COP also set to 0 for outputs)
            self.shr = 0
            self.capacity = 0
            self.eir = 0

        mult = 1 if self.is_heater else -1
        heat_gain = mult * self.capacity  # Heat gain in W, positive=heat, negative=cool
        self.fan_power = self.fan_power_rated if 'On' in self.mode else 0

        # Delivered heat: heat gain subject to SHR, both heat gain and fan power subject to DSE
        self.delivered_heat = (heat_gain * self.shr + self.fan_power) * self.duct_dse
        self.sensible_gain = self.delivered_heat  # exactly the same
        self.latent_gain = heat_gain * (1 - self.shr) * self.duct_dse

        # Total power: includes fan power when on
        power_kw = abs(heat_gain) / 1000 * self.eir
        if self.is_gas:
            self.gas_therms_per_hour = Units.kWh2therms(power_kw)
            if self.is_electric:
                self.electric_kw = self.fan_power / 1000
        elif self.is_electric:
            self.electric_kw = power_kw + self.fan_power / 1000

        # reduce delivered heat (only for results) and power output based on space fraction
        # Note: sensible/latent gains to envelope are not updated
        self.delivered_heat *= self.space_fraction
        self.electric_kw *= self.space_fraction
        self.gas_therms_per_hour *= self.space_fraction

        # update previous indoor temperature
        self.temp_indoor_prev = schedule['Indoor']

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            mult = 1 if self.is_heater else -1
            return {self.end_use: {
                # 'T Indoor': temp_indoor,
                'T Setpoint': self.temp_setpoint,
                'T Upper Limit': self.temp_setpoint + self.temp_deadband / 2,
                'T Lower Limit': self.temp_setpoint - self.temp_deadband / 2,
                'Power': self.electric_kw,
                'Rated COP': 1 / self.eir_rated,
                'COP': 1 / self.eir if self.eir else 0,  # Note: using biquadratic for dynamic HVAC equipment
                'Capacity': Units.W2Ton(self.capacity),  # Note: using biquadratic for dynamic HVAC equipment
                'Mode': mult * int('On' in self.mode),
            }}
        else:
            if verbosity >= 3:
                # Note: using end use, not equipment name, for all results
                results[self.end_use + ' Delivered (kW)'] = abs(self.delivered_heat) / 1000
            if verbosity >= 6:
                results.update({
                    self.end_use + ' Gross Power (kW)'.format(
                        self.hvac_type.capitalize()): self.electric_kw - self.fan_power / 1000 * self.space_fraction,
                    self.end_use + ' Fan Power (kW)': self.fan_power / 1000 * self.space_fraction,
                    self.end_use + ' COP (-)': 1 / self.eir if self.eir else 0,
                    self.end_use + ' SHR (-)': self.shr,
                    self.end_use + ' Capacity (tons)': Units.W2Ton(self.capacity),
                })
        return results


class Heater(HVAC):
    hvac_type = 'heat'
    end_use = 'HVAC Heating'


class Cooler(HVAC):
    hvac_type = 'cool'
    end_use = 'HVAC Cooling'


class GasFurnace(Heater):
    name = 'Gas Furnace'
    is_electric = False
    is_gas = True


class IdealHVAC(HVAC):
    """
    HVAC Equipment Class using ideal capacity algorithm. This uses the envelope model to determine the exact HVAC
     capacity to maintain the setpoint temperature. It does not account for heat gains from other equipment in the
     same time step.
    """
    no_fan_power = True

    def __init__(self, envelope_model, **kwargs):
        super().__init__(**kwargs)

        # Building envelope parameters - required for calculating ideal capacity
        self.envelope_model = envelope_model
        self.envelope_inputs = {}

        # Fan power is proportional to the capacity for ideal equipment
        if self.no_fan_power:
            self.fan_power_rated = 0
        self.fan_power_ratio = self.fan_power_rated / self.capacity_rated

        # Equipment turns off if ideal capacity is below the minimum
        self.capacity_min = kwargs.get(self.hvac_type + 'ing minimum capacity (W)', 0)  # in W

    def run_duty_cycle_control(self, schedule, ext_control_args):
        duty_cycles = ext_control_args.get('{}ing Duty Cycle'.format(self.hvac_type.capitalize()))
        if duty_cycles is None:
            raise EquipmentException('Unknown duty cycle control for {}: {}'.format(self.name, ext_control_args))

        # set capacity to constant value based on duty cycle
        assert isinstance(duty_cycles, (int, float)) and 0 <= duty_cycles <= 1
        self.capacity = duty_cycles * self.capacity_rated
        if self.capacity < self.capacity_min:
            self.capacity = 0

        # set to first mode (default) if duty cycle > 0, otherwise 'Off'
        return self.modes[0] if self.capacity > 0 else 'Off'

    def update_internal_control(self, schedule):
        # Update setpoint from schedule file
        self.temp_setpoint = schedule['{}ing_setpoint'.format(self.hvac_type)]

        # run ideal capacity calculation here, just to determine mode
        # FUTURE: capacity update is done twice per loop, could but updated to improve speed
        self.update_capacity(schedule)

        # set to first mode (default) if delivered heat > 0, otherwise 'Off'
        return self.modes[0] if self.capacity > 0 else 'Off'

    def update_capacity(self, schedule):
        # Update capacity using ideal algorithm - maintains setpoint exactly
        x_desired = self.temp_setpoint

        # Get all envelope inputs for this time step, includes other equipment, occupancy, solar gains, infiltration
        to_model = self.envelope_model.get_model_inputs(schedule['to_envelope'].copy(), schedule)
        _ = to_model.pop('H_LIV_latent')

        # Solve for desired H_LIV, subtracting external gains
        self.envelope_model.update_inputs(to_model)
        ext_gains = self.envelope_model.get_input('H_' + self.air_node)
        h_desired = self.envelope_model.solve_for_input('T_' + self.air_node, 'H_' + self.air_node,
                                                        x_desired)  # in W
        h_hvac_net = h_desired - ext_gains

        # Account for fan power, SHR and duct DSE - slightly different for heating/cooling
        if self.is_heater:
            h_hvac = h_hvac_net / self.duct_dse / (self.shr + self.fan_power_ratio)
        else:
            h_hvac = - h_hvac_net / self.duct_dse / (self.shr - self.fan_power_ratio)

        # force min capacity <= capacity <= rated capacity. If ideal capacity is out of bounds, setpoint won't be met
        self.capacity = np.clip(h_hvac, self.capacity_min, self.capacity_rated)

        # Update fan power as proportional to capacity
        self.fan_power = self.capacity * self.fan_power_ratio


class IdealHeater(IdealHVAC, Heater):
    name = 'Ideal Heater'


class IdealCooler(IdealHVAC, Cooler):
    name = 'Ideal Cooler'


class ElectricFurnace(IdealHeater):
    name = 'Electric Furnace'


class ElectricBoiler(IdealHeater):
    name = 'Electric Boiler'


class ElectricBaseboard(IdealHeater):
    name = 'Electric Baseboard'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # force duct dse to 1
        self.duct_dse = 1


class GasBoiler(IdealHeater):
    name = 'Gas Boiler'
    is_electric = False
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Boiler specific inputs
        self.condensing = self.eir_rated < 1 / 0.9  # Condensing if efficiency (AFUE) > 90%

        if self.condensing:
            self.outlet_temp = 65.56  # outlet_water_temp [C] (150 F)
            self.efficiency_coeff = np.array([1.058343061, -0.052650153, -0.0087272,
                                              -0.001742217, 0.00000333715, 0.000513723], dtype=float)
        else:
            self.outlet_temp = 82.22  # self.outlet_water_temp [C] (180F)
            self.efficiency_coeff = np.array([1.111720116, 0.078614078, -0.400425756, 0, -0.000156783, 0.009384599,
                                              0.234257955, 0.00000132927, -0.004446701, -0.0000122498], dtype=float)

    def update_eir(self, schedule):
        # update EIR based on part load ratio, input/output temperatures
        plr = self.capacity / self.capacity_rated  # part-load-ratio
        t_in = schedule['Indoor']
        t_out = self.outlet_temp
        if self.condensing:
            eff_var = np.array([1, plr, plr ** 2, t_in, t_in ** 2, plr * t_in], dtype=float)
            eff_curve_output = np.dot(eff_var, self.efficiency_coeff)
        else:
            eff_var = np.array([1, plr, plr ** 2, t_out, t_out ** 2, plr * t_out, plr ** 3, t_out ** 3,
                                plr ** 2 * t_out, plr * t_out ** 2], dtype=float)
            eff_curve_output = np.dot(eff_var, self.efficiency_coeff)
        self.eir = self.eir_rated / eff_curve_output


class DynamicHVAC(HVAC):
    """
    HVAC Equipment Class using dynamic capacity algorithm. This uses a biquadratic model to update the EIR, SHR,
    and capacity at each time step. For more details, see:
     D. Cutler (2013) Improved Modeling of Residential Air Conditioners and Heat Pumps for Energy Calculations
     https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/modeling_ac_heatpump.pdf
     Section 2.2.1, Equations 7-8, 11-13
    """

    def __init__(self, speed_type='Single', control_type='Time', **kwargs):
        # Dynamic HVAC Parameters
        if speed_type not in SPEED_TYPES:
            raise EquipmentException('Invalid Speed Type for {}: {}'.format(self.name, speed_type))
        self.speed_type = speed_type
        self.control_type = control_type  # 'Time' or 'Setpoint', for 2-speed control only

        # Load biquadratic parameters - only keep those with the correct speed type
        biquadratic_file = kwargs.get('biquadratic_file', 'biquadratic_parameters.csv')
        biquad_params = self.initialize_parameters(biquadratic_file, val_col=None, **kwargs).to_dict()
        self.biquad_params = {
            col.split('_')[1]: {'eir_t': np.array([val[x + '_eir_T'] for x in 'abcdef'], dtype=float),
                                'eir_ff': np.array([val[x + '_EIR_FF'] for x in 'abc'], dtype=float),
                                'cap_t': np.array([val[x + '_Qcap_T'] for x in 'abcdef'], dtype=float),
                                'cap_ff': np.array([val[x + '_Qcap_FF'] for x in 'abc'], dtype=float),
                                'min_Twb': val.get('min_Twb', -100),
                                'max_Twb': val.get('max_Twb', 100),
                                'min_Tdb': val.get('min_Tdb', -100),
                                'max_Tdb': val.get('max_Tdb', 100)}
            for col, val in biquad_params.items() if speed_type == col.split('_')[0]
        }

        if not self.biquad_params:
            raise EquipmentException('Biquadratic parameters not found for {} speed {}'.format(speed_type, self.name))

        # Update HVAC equipment modes
        self.modes = self.define_modes()

        super().__init__(**kwargs)

        self.flow_fraction = kwargs.get('Flow_Fraction', 1.0)  # Flow fraction (flow rate/rated flow rate)

        # Initialize EIR and Capacity
        self.eir_ratio = 1
        self.capacity_ratio = 1

        # 2-speed only: force low speed minimum time to be at least 5 minutes
        if self.speed_type == 'Double':
            low_mode = self.modes[0]  # first mode corresponds to lowest HP mode
            self.min_time_in_mode[low_mode] = max(self.min_time_in_mode[low_mode], dt.timedelta(minutes=5))

    def define_modes(self):
        return ['On - ' + speed for speed in self.biquad_params.keys()] + ['Off']

    def run_duty_cycle_control(self, schedule, ext_control_args):
        # Number of duty cycles depends on speed type
        if self.speed_type == 'Single':
            return super().run_duty_cycle_control(schedule, ext_control_args)
        elif self.speed_type == 'Double':
            duty_cycles = [ext_control_args.get('{}ing Duty Cycle 1'.format(self.hvac_type.capitalize())),
                           ext_control_args.get('{}ing Duty Cycle 2'.format(self.hvac_type.capitalize())),
                           ext_control_args.get('{}ing Duty Cycle Off'.format(self.hvac_type.capitalize()), 0)]

            mode_priority = self.calculate_mode_priority(*duty_cycles)
            thermostat_mode = self.run_thermostat_control(schedule)

            if thermostat_mode in mode_priority and not self.ext_ignore_thermostat:
                return thermostat_mode
            else:
                return mode_priority[0]  # take highest priority mode (usually current mode)

    def run_two_speed_control(self, schedule):
        mode = super().run_thermostat_control(schedule)
        mode_on = mode == 'On' or (mode is None and 'On' in self.mode)

        if self.control_type == 'Time':
            # Time-based 2-speed HVAC control: High speed turns on if temp continues to drop (for heating)
            if mode_on:
                mult = 1 if self.is_heater else -1
                if self.mode == 'Off':
                    return 'On - 1'
                elif mult * (schedule['Indoor'] - self.temp_indoor_prev) < 0:
                    return 'On - 2'
                else:
                    return 'On - 1'
            else:
                return 'Off'
        elif self.control_type == 'Setpoint':
            # Setpoint-based 2-speed HVAC control: High speed uses setpoint difference of deadband / 2 (overlapping)
            new_schedule = schedule.copy()
            if self.is_heater:
                new_schedule['heating_setpoint'] -= self.temp_deadband / 2
            else:
                new_schedule['cooling_setpoint'] += self.temp_deadband / 2
            mode2 = super().run_thermostat_control(new_schedule)
            mode2_on = mode2 == 'On' or (mode2 is None and 'On - 2' in self.mode)

            if mode2_on:
                return 'On - 2'
            elif mode_on:
                return 'On - 1'
            else:
                return 'Off'
        elif self.control_type == 'Time2':
            # Old time-based 2-speed HVAC control
            if mode_on:
                if self.mode == 'Off':
                    return 'On - 1'
                else:
                    return 'On - 2'
            else:
                return 'Off'
        else:
            raise EquipmentException('Unknown control type for {}: {}'.format(self.name, self.control_type))

    def run_thermostat_control(self, schedule):
        if self.speed_type == 'Single':
            # Run regular thermostat control
            mode = super().run_thermostat_control(schedule)
            if mode == 'On':
                mode = 'On - 1'
            return mode

        elif self.speed_type == 'Double':
            return self.run_two_speed_control(schedule)

        elif self.speed_type == 'Variable':
            raise EquipmentException('Variable speed equipment must implement ideal capacity algorithm.'
                                     'Rename equipment to "Ideal ASHP Heater" or "Ideal ASHP Cooler"')

    def calculate_biquadratic_ratio(self, schedule, param, speed):
        # runs biquadratic equation for EIR or capacity given the speed
        t_in_db = schedule['Indoor']
        t_in_wb = self.coil_input_air_wb
        t_ext_db = schedule['ambient_dry_bulb']

        # use wet bulb for cooling, dry bulb for heating
        t_in = t_in_db if self.is_heater else t_in_wb

        # get biquadratic parameters for current speed
        params = self.biquad_params[speed]

        # clip temperatures to stay within bounds
        t_in = np.clip(t_in, params['min_Twb'], params['max_Twb'])
        t_ext_db = np.clip(t_ext_db, params['min_Tdb'], params['max_Tdb'])

        # create vectors based on temperature and flow fraction
        t_list = np.array([1, t_in, t_in ** 2, t_ext_db, t_ext_db ** 2, t_in * t_ext_db], dtype=float)
        ff_list = np.array([1, self.flow_fraction, self.flow_fraction ** 2], dtype=float)

        t_ratio = np.dot(t_list, params[param + '_t'])
        ff_ratio = np.dot(ff_list, params[param + '_ff'])
        ratio = t_ratio * ff_ratio
        return ratio

    def update_capacity(self, schedule):
        # Update capacity using biquadratic model
        # TODO: need to know which speed to use, can be issue with ideal equipment
        speed = self.mode.split(' - ')[1] if ' - ' in self.mode else '1'
        self.capacity_ratio = self.calculate_biquadratic_ratio(schedule, param='cap', speed=speed)
        self.capacity = self.capacity_rated * self.capacity_ratio

    def update_eir(self, schedule):
        # Update EIR using biquadratic model
        # TODO: need to know which speed to use, can be issue with ideal equipment
        speed = self.mode.split(' - ')[1] if ' - ' in self.mode else '1'
        self.eir_ratio = self.calculate_biquadratic_ratio(schedule, param='eir', speed=speed)
        self.eir = self.eir_rated * self.eir_ratio

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)

        if not to_ext and verbosity >= 6:
            results.update({
                self.end_use + ' EIR Ratio (-)': self.eir_ratio,
                self.end_use + ' Capacity Ratio (-)': self.capacity_ratio,
            })
        return results


class AirConditioner(DynamicHVAC, Cooler):
    name = 'Air Conditioner'

    def calculate_power_and_heat(self, schedule):
        super().calculate_power_and_heat(schedule)

        # add 20W of fan power when off and outdoor temp < 55F
        # no impact on sensible heat for now
        # TODO: add input for crankcase heater rather than always on or off
        # if self.mode == 'Off' and schedule['ambient_dry_bulb'] < Units.F2C(55):
        #    self.electric_kw += 0.020 * self.space_fraction


class ASHPCooler(AirConditioner):
    name = 'ASHP Cooler'


class RoomAC(AirConditioner):
    name = 'Room AC'

    def __init__(self, **kwargs):
        if kwargs.get('speed_type', 'Single') != 'Single':
            raise EquipmentException('No model for multi-speed {}'.format(self.name))
        super().__init__(**kwargs)


class IdealDynamicHVAC(IdealHVAC, DynamicHVAC):
    """
    Class that uses the ideal algorithm to determine HVAC capacity, and biquadratic equations to determine HVAC
    efficiency. Variable speed air source heat pump model currently uses this implementation.
    """
    def define_modes(self):
        # keep original modes (On and Off)
        return self.modes

    def update_eir(self, schedule):
        # TODO: implement interpolation of biquadratic equations to calculate EIR
        #  - Note: self.capacity and self.capacity_ratio are already defined

        eir_ratio_1 = self.calculate_biquadratic_ratio(schedule, param='eir', speed='1')
        # eir_ratio_2 = self.calculate_biquadratic_ratio(schedule, param='eir', speed='2')
        # eir_ratio_3 = self.calculate_biquadratic_ratio(schedule, param='eir', speed='3')
        # eir_ratio_4 = self.calculate_biquadratic_ratio(schedule, param='eir', speed='4')

        self.eir_ratio = eir_ratio_1
        self.eir = self.eir_rated * self.eir_ratio


class IdealASHPHeater(IdealDynamicHVAC, Heater):
    name = 'Ideal ASHP Heater'
    folder_name = 'ASHP Heater'
    no_fan_power = False


class IdealASHPCooler(IdealDynamicHVAC, Cooler):
    name = 'Ideal ASHP Cooler'
    folder_name = 'ASHP Cooler'
    no_fan_power = False


class IdealRoomAC(IdealASHPCooler):
    name = 'Ideal Room AC'
    folder_name = 'Room AC'


class HeatPumpHeater(DynamicHVAC, Heater):
    name = 'Heat Pump Heater'
    folder_name = 'ASHP Heater'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Defrost Parameters
        self.defrost = False
        self.power_defrost = 0
        self.defrost_power_mult = 1

        # Electric heating element parameters
        self.outdoor_temp_limit = kwargs['supplemental heater cut in temp (C)']

    def run_thermostat_control(self, schedule):
        new_mode = super().run_thermostat_control(schedule)

        # Force off if outdoor temp is very cold
        t_ext_db = schedule['ambient_dry_bulb']
        if t_ext_db < self.outdoor_temp_limit:
            return 'Off'
        else:
            return new_mode

    def update_capacity(self, schedule):
        # Update capacity if defrost is required
        super().update_capacity(schedule)

        t_ext_db = schedule['ambient_dry_bulb']
        rh_ext = schedule['ambient_humidity']
        pres_ext = schedule['ambient_pressure']

        # Based on EnergyPlus Engineering Reference, Frost Adjustment Factors, for on demand, reverse cycle defrost
        self.defrost = t_ext_db < 4.4445
        if self.defrost:
            # Calculate reduced capacity
            T_coil_out = 0.82 * t_ext_db - 8.589
            omega_out = PsyCalc.w_fT_R_P_SI(t_ext_db, rh_ext, pres_ext)
            omega_sat_coil = PsyCalc.w_fT_Twb_P_SI(T_coil_out, T_coil_out, pres_ext)  # saturation humidity ratio
            # TODO: Jeff: seems like omega_out < omega_sat_coil for t_ext_db=0C, can you review?
            delta_omega_coil_out = max(0.000001, omega_out - omega_sat_coil)
            defrost_time_frac = 1.0 / (1 + (0.01446 / delta_omega_coil_out))
            defrost_capacity_mult = 0.875 * (1 - defrost_time_frac)
            self.defrost_power_mult = 0.954 / 0.875  # increase in power relative to the capacity

            q_defrost = 0.01 * defrost_time_frac * (7.222 - t_ext_db) * (self.capacity / 1.01667)
            self.capacity = self.capacity * defrost_capacity_mult - q_defrost

            # Calculate additional power and EIR
            defrost_eir_temp_mod_frac = 0.1528  # in kW
            self.power_defrost = defrost_eir_temp_mod_frac * (self.capacity / 1.01667) * defrost_time_frac
        else:
            self.defrost_power_mult = 0
            self.power_defrost = 0

    def update_eir(self, schedule):
        # Update EIR from defrost. Assumes update_capacity is already run
        super().update_eir(schedule)
        if self.defrost:
            self.eir = (self.eir * self.capacity * self.defrost_power_mult + self.power_defrost) / self.capacity


class ASHPHeater(HeatPumpHeater):
    """
    Heat pump heater with a backup electric resistance element
    """
    name = 'ASHP Heater'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # backup element parameters
        self.supplemental_capacity_rated = kwargs['supplemnetal heating capacity (W)']
        self.supplemental_eir_rated = 1

    def define_modes(self):
        # add ER modes to list of modes
        modes = ['On - ' + speed for speed in self.biquad_params.keys()]
        return ['HP ' + mode for mode in modes] + ['HP and ER ' + mode for mode in modes] + ['ER On', 'Off']

    def run_duty_cycle_control(self, schedule, ext_control_args):
        # If duty cycles exist, combine duty cycles for HP and ER modes
        er_duty_cycle = ext_control_args.get('ER Duty Cycle', 0)
        if self.speed_type == 'Single':
            hp_duty_cycle = ext_control_args.get('Heating Duty Cycle', 0)
            duty_cycles = [hp_duty_cycle, min(hp_duty_cycle, er_duty_cycle), er_duty_cycle,
                           1 - hp_duty_cycle - er_duty_cycle]  # 4 modes for single speed equipment
        elif self.speed_type == 'Double':
            hp1_duty_cycle = ext_control_args.get('Heating Duty Cycle 1', 0)
            hp2_duty_cycle = ext_control_args.get('Heating Duty Cycle 2', 0)
            duty_cycles = [hp1_duty_cycle, hp2_duty_cycle, min(hp1_duty_cycle, er_duty_cycle),
                           min(hp2_duty_cycle, er_duty_cycle), er_duty_cycle,
                           1 - hp1_duty_cycle - hp2_duty_cycle - er_duty_cycle]  # 6 modes for 2 speed equipment
        else:
            raise EquipmentException('Unknown speed type for {}: {}'.format(self.name, self.speed_type))

        mode_priority = self.calculate_mode_priority(*duty_cycles)
        thermostat_mode = self.run_thermostat_control(schedule)

        if thermostat_mode in mode_priority and not self.ext_ignore_thermostat:
            mode = thermostat_mode
        else:
            mode = mode_priority[0]  # take highest priority mode (usually current mode)
        if mode is None:
            mode = self.mode

        if 'HP' in mode:
            speed = mode.split(' - ')[1]
            if 'ER' in mode:
                # update HP only and ER only counters
                self.ext_mode_counters['HP On - ' + speed] += self.time_res
                self.ext_mode_counters['ER On'] += self.time_res
            else:
                # update HP+ER counter
                self.ext_mode_counters['HP and ER On - ' + speed] = max(self.ext_mode_counters[mode] + self.time_res,
                                                                        self.ext_mode_counters['ER On'])
        elif 'ER' in mode:
            # update all HP+ER counters
            for s in self.biquad_params.keys():
                self.ext_mode_counters['HP and ER On - ' + s] = max(self.ext_mode_counters['HP On - ' + s],
                                                                    self.ext_mode_counters['ER On'] + self.time_res)
        return mode

    def run_thermostat_control(self, schedule):
        t_ext_db = schedule['ambient_dry_bulb']

        # run thermostat control for HP
        mode = DynamicHVAC.run_thermostat_control(self, schedule)

        # run thermostat control for ER element
        new_schedule = schedule.copy()
        # TODO: add option to keep setpoint as is, e.g. when using external control
        new_schedule['heating_setpoint'] -= self.temp_deadband
        er_mode = HVAC.run_thermostat_control(self, new_schedule)

        if t_ext_db < self.outdoor_temp_limit:
            # at low outdoor temps: force HP off, use normal mode for ER
            return 'ER On' if 'On' in mode else mode
        else:
            # determine next mode - either HP or HP+ER or neither
            hp_on = (mode is not None and 'On' in mode) or (mode is None and 'HP' in self.mode)
            er_on = er_mode == 'On' or (er_mode is None and 'ER' in self.mode)
            if er_on:
                if hp_on:
                    return 'HP and ER ' + mode
                else:
                    return 'ER On'
            else:
                if mode is not None and 'On' in mode:
                    return 'HP ' + mode
                else:
                    return mode

    def update_capacity(self, schedule):
        if 'HP and ER' in self.mode:
            super().update_capacity(schedule)
            self.capacity += self.supplemental_capacity_rated
        elif 'HP' in self.mode:
            super().update_capacity(schedule)
        elif 'ER' in self.mode:
            self.capacity = self.supplemental_capacity_rated
        else:
            raise EquipmentException('Unknown mode for {}: {}'.format(self.name, self.mode))

    def update_eir(self, schedule):
        if 'HP and ER' in self.mode:
            super().update_eir(schedule)
            # EIR is a weighted average of HP and ER EIRs
            hp_capacity = self.capacity - self.supplemental_capacity_rated
            self.eir = (hp_capacity * self.eir + self.supplemental_capacity_rated * self.supplemental_eir_rated) / (
                    hp_capacity + self.supplemental_capacity_rated)
        elif 'HP' in self.mode:
            super().update_eir(schedule)
        elif 'ER' in self.mode:
            self.eir = self.supplemental_eir_rated
        else:
            raise EquipmentException('Unknown mode for {}: {}'.format(self.name, self.mode))

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            results.update({
                'HP Mode': int('HP' in self.mode),
                'ER Mode': int('ER' in self.mode),
                'ER Capacity': Units.W2Ton(self.supplemental_capacity_rated),
            })
        return results
