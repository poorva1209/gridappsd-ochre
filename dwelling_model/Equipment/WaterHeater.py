# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:24:32 2018

@author: kmckenna, mblonsky
"""
import numpy as np

from dwelling_model.Equipment import Equipment
from dwelling_model.Models import OneNodeWaterModel, StratifiedWaterModel, IdealWaterModel
from dwelling_model.Models.Envelope import MAIN_NODES as ENVELOPE_NODES
from dwelling_model import Units

# Water Constants
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K


class TankWaterHeater(Equipment):
    name = 'Water Heater'
    model_class = StratifiedWaterModel
    end_use = 'Water Heating'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create water model
        self.model = self.model_class(**kwargs)
        self.upper_node = '3' if self.model.n_nodes >= 12 else '1'
        self.lower_node = '10' if self.model.n_nodes >= 12 else str(self.model.n_nodes)

        # Control parameters
        self.upper_threshold_temp = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        deadband_temp = kwargs.get('deadband (delta C)', 5.56)
        self.lower_threshold_temp = self.upper_threshold_temp - deadband_temp
        self.upper_threshold_temp = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        self.max_temp = Units.F2C(kwargs.get('max tank temperature (F)', 140))

        # Other parameters
        self.tempered_draw_temp = Units.F2C(kwargs.get('mixed delivery temperature (F)', 110))
        self.efficiency = kwargs.get('eta_c', 1)
        self.delivered_heat = 0  # heat delivered to the tank, in W
        if kwargs.get('water heater location') == 'living':
            # TODO: update for other locations
            self.air_node = 'LIV'

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces WH off)
        # - Setpoint: Updates setpoint temperature from the dwelling schedule (in C)
        #   - Note: Setpoint will not reset back to original value
        # - Deadband: Updates deadband temperature (in C)
        #   - Note: Deadband will not reset back to original value
        # - Duty Cycle: Forces WH on for fraction of external time step (as fraction [0,1])
        #   - If 0 < Duty Cycle < 1, the equipment will cycle once every 2 external time steps
        #   - For HPWH: Can supply HP and ER duty cycles
        #   - Note: does not use clock on/off time

        # If load fraction = 0, force off
        load_fraction = ext_control_args.get('Load Fraction', 1)
        if load_fraction == 0:
            return 'Off'
        elif load_fraction != 1:
            raise Exception("{} can't handle non-integer load fractions".format(self.name))

        ext_setpoint = ext_control_args.get('Setpoint')
        if ext_setpoint is not None:
            if ext_setpoint > self.max_temp:
                print('WARNING: {} setpoint cannot exceed {}C. Setting setpoint to maximum value.'
                      .format(self.name, self.max_temp))
                ext_setpoint = self.max_temp

            # keep deadband the same
            self.lower_threshold_temp += ext_setpoint - self.upper_threshold_temp
            self.upper_threshold_temp = ext_setpoint

        ext_db = ext_control_args.get('Deadband')
        if ext_db is not None:
            self.lower_threshold_temp = self.upper_threshold_temp - ext_db

        duty_cycles = ext_control_args.get('Duty Cycle')
        if duty_cycles is None:
            return self.update_internal_control(schedule)

        # Force off if temperature exceeds maximum, and print warning
        t_tank = self.model.get_state('T_' + self.upper_node)
        if t_tank > self.max_temp:
            print('WARNING: {} temperature over maximum temperature ({}C), forcing off'
                  .format(self.name, self.max_temp))
            return 'Off'

        # Parse duty cycles into list for each mode
        if isinstance(duty_cycles, (int, float)):
            duty_cycles = [duty_cycles]
        if len(duty_cycles) == len(self.modes) - 2:
            # copy duty cycle for Upper On and Lower On, and calculate Off duty cycle
            duty_cycles.append(duty_cycles[-1])
            duty_cycles.append(1 - sum(duty_cycles[:-1]))

        # Use internal mode if available, otherwise use mode with highest priority
        mode_priority = self.calculate_mode_priority(*duty_cycles)
        internal_mode = self.update_internal_control(schedule)
        if internal_mode is None:
            internal_mode = self.mode
        if internal_mode in mode_priority:
            return internal_mode
        else:
            return mode_priority[0]  # take highest priority mode (usually current mode)

    def update_internal_control(self, schedule):
        t_tank = self.model.get_state('T_' + self.lower_node)  # use lower node for gas WH, not for ERWH

        if t_tank < self.lower_threshold_temp:
            return 'On'
        if self.mode == 'On' and t_tank > self.upper_threshold_temp:
            return 'Off'

    def calculate_power_and_heat(self, schedule):
        raise NotImplementedError()

    def update_model(self, to_model, schedule):
        # run model update, assumes water heater is in main indoor node, assumes no latent gains
        heat_loss = self.model.update(to_model, self.tempered_draw_temp, schedule)
        self.sensible_gain += heat_loss

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            t_avg = np.dot(self.model.vol_fractions, self.model.x)  # weighted average temperature
            return {self.name: {'T Average': t_avg,
                                'T Upper Node': self.model.get_state('T_' + self.upper_node),
                                'T Lower Node': self.model.get_state('T_' + self.lower_node),
                                'T Upper Limit': self.upper_threshold_temp,
                                'T Lower Limit': self.lower_threshold_temp,
                                'Is On': 'On' in self.mode,
                                'Tank Volume': self.model.volume}}
        else:
            if verbosity >= 3:
                # Note: using end use, not equipment name, for all results
                results.update({self.end_use + ' Delivered (kW)': self.delivered_heat / 1000})
            if verbosity >= 6:
                cop = self.delivered_heat / (self.electric_kw * 1000) if self.electric_kw > 0 else 0
                results.update({self.end_use + ' COP (-)': cop})
            results.update(self.model.generate_results(verbosity, to_ext))
        return results


class ElectricResistanceWaterHeater(TankWaterHeater):
    name = 'Electric Resistance Water Heater'
    modes = ['Lower On', 'Upper On', 'Off']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Power parameters (in kW)
        self.upper_power = kwargs['rated input power (W)'] / 1000
        self.lower_power = kwargs['rated input power (W)'] / 1000

    def update_external_control(self, schedule, ext_control_args):
        mode = super().update_external_control(schedule, ext_control_args)
        t_upper = self.model.get_state('T_' + self.upper_node)

        # If duty cycle forces WH on, may need to swap to lower element
        if mode == 'Upper On' and t_upper > self.upper_threshold_temp:
            mode = 'Lower On'

        # If mode is ER, add time to both mode_counters
        if mode == 'Upper On':
            self.ext_mode_counters['Lower On'] += self.time_res
        if mode == 'Lower On':
            self.ext_mode_counters['Upper On'] += self.time_res

        return mode

    def update_internal_control(self, schedule):
        t_upper = self.model.get_state('T_' + self.upper_node)
        t_lower = self.model.get_state('T_' + self.lower_node)

        # Note: Upper unit gets priority over lower unit
        if t_upper < self.lower_threshold_temp:
            return 'Upper On'
        if self.mode == 'Upper On' and t_upper > self.upper_threshold_temp:
            return 'Off'

        if self.mode == 'Off' and t_lower < self.lower_threshold_temp:
            return 'Lower On'
        if self.mode == 'Lower On' and t_lower > self.upper_threshold_temp:
            return 'Off'

    def calculate_power_and_heat(self, schedule):
        to_model = {}
        if self.mode == 'Upper On':
            self.electric_kw = self.upper_power
            self.delivered_heat = self.upper_power * self.efficiency * 1000
            self.sensible_gain = self.upper_power * (1 - self.efficiency) * 1000  # in W
            to_model['H_' + self.upper_node] = self.delivered_heat
        elif self.mode == 'Lower On':
            self.electric_kw = self.lower_power
            self.delivered_heat = self.lower_power * self.efficiency * 1000
            self.sensible_gain = self.lower_power * (1 - self.efficiency) * 1000  # in W
            to_model['H_' + self.lower_node] = self.delivered_heat
        else:
            self.electric_kw = 0
            self.delivered_heat = 0
            self.sensible_gain = 0

        self.latent_gain = 0
        self.update_model(to_model, schedule)


class HeatPumpWaterHeater(ElectricResistanceWaterHeater):
    name = 'Heat Pump Water Heater'
    modes = ['Heat Pump On', 'Upper On', 'Lower On', 'Off']

    def __init__(self, hp_only_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.lower_node = '9' if self.model.n_nodes >= 12 else '2'

        # Control parameters
        self.hp_only_mode = hp_only_mode
        self.t_upper_prev = self.model.get_state('T_' + self.upper_node)
        self.upper_threshold_temp_prev = self.upper_threshold_temp

        # Nominal COP based on simulation of the UEF test procedure at varying COPs
        self.cop_nominal = kwargs.get('HPWH COP', 1.174536058 * (0.60522 + kwargs['EF']) / 1.2101)
        self.cop = self.cop_nominal

        # Heat pump capacity - hardcoded for now
        self.hp_power_nominal = 979.0  # W
        self.hp_power = self.hp_power_nominal

        # Dynamic capacity coefficients
        # curve format: [const, f(Twb,amb), f(Twb,amb)**2, f(Tw],f(Tw)**2, f(Twb,amb)*f(Tw)]
        self.cop_coeff = [0.563, 0.0437, 0.000039, 0.0055, -0.000148, -0.000145]
        self.hp_power_coeff = [1.1332, 0.063, -0.0000979, -0.00972, -0.0000214, -0.000686]

        # Other HP coefficients
        self.shr_nominal = kwargs.get('HPWH SHR', 0.98)  # unitless
        self.parasitic_power = kwargs.get('HPWH Parasitics (W)', 3.0)  # Standby power in W
        self.fan_power = kwargs.get('HPWH Fan Power (W)', 0.0462 * 181)  # in W

    def update_external_control(self, schedule, ext_control_args):
        if any([dc in ext_control_args for dc in ['HP Duty Cycle', 'ER Duty Cycle']]):
            # Add HP duty cycle to ERWH control
            duty_cycles = [ext_control_args.get('HP Duty Cycle', 0),
                           ext_control_args.get('ER Duty Cycle', 0) if not self.hp_only_mode else 0]
            ext_control_args['Duty Cycle'] = duty_cycles

        return super().update_external_control(schedule, ext_control_args)

    def update_internal_control(self, schedule):
        """
        Uses AO Smith HPWH logic for 12 node tank

        Various other HPWH control logic for 2 node tank described below

        # Use different lower threshold temp based on HPWH specific deadband
        temp_deadband = (self.upper_threshold_temp * 0.3) - 12.
        lower_threshold_temp = self.upper_threshold_temp - temp_deadband

        on = 'On' in self.mode
        if self.hp_only_mode:
            if t_upper < (self.upper_threshold_temp - 2) and t_lower < (self.upper_threshold_temp - 20) and \
                    not on or (t_upper < (self.upper_threshold_temp - 5.56) and not on):
                mode = 'Heat Pump On'
            elif on and t_lower < self.upper_threshold_temp:
                # Keep the HP on until the lower node hits set point
                mode = 'Heat Pump On'
            else:
                mode = 'Off'
        else:
            # old code
            # delta_t = (t_upper - self.t_upper_prev) / (self.time_res.total_seconds() / 60)  # degrees C/min
            # if t_upper < lower_threshold_temp and not on:  # Turn on HP 1
            #     mode = 'Heat Pump On'
            # elif t_upper < (self.upper_threshold_temp - 0.5) and -0.75 < delta_t and not on:  # Turn on HP 2
            #     mode = 'Heat Pump On'
            # elif t_upper < (self.upper_threshold_temp - 0.5) and -2.0 < delta_t < -0.75 and \
            #         (not on or self.mode == 'Heat Pump On'):  # Turn on LE
            #     mode = 'Lower On'
            # elif t_upper < self.upper_threshold_temp - 0.5 and delta_t < -2.0:  # Turn on UE
            #     mode = 'Upper On'
            # elif self.mode == 'Upper On' and t_upper < self.upper_threshold_temp and \
            #         t_upper < self.upper_threshold_temp - 4.0:  # Switch from UE to LE
            #     mode = 'Lower On'
            # elif self.mode == 'Heat Pump On' and t_upper < self.upper_threshold_temp:  # Keep the HP on
            #     mode = 'Heat Pump On'
            # elif t_upper > self.upper_threshold_temp:  # Turn everything off
            #     mode = 'Off'
            # else:
            #     mode = None

            # new code
            # if (self.t_upper_prev > 50 and (self.upper_threshold_temp - self.upper_threshold_temp_prev) > 3) or (
            #         (t_upper < (self.upper_threshold_temp - 2)) and (
            #         t_lower < (self.upper_threshold_temp - 20)) and not on):  # Turn on HP
            #     mode = 'Heat Pump On'
            # elif (t_upper < self.upper_threshold_temp) and (t_lower < (self.upper_threshold_temp - 40)):
            #     mode = 'Upper On'
            # elif (t_upper >= self.upper_threshold_temp) and self.mode == 'Upper On':
            #     mode = 'Lower On'
            # elif (t_upper < self.upper_threshold_temp) and ((t_lower < (self.upper_threshold_temp - 35)) or (
            #         t_upper < (self.upper_threshold_temp - 3.6) and t_lower < (
            #         self.upper_threshold_temp - 30))):  # Turn on lower element
            #     mode = 'Lower On'
            # elif self.mode == 'Heat Pump On' and t_lower < self.upper_threshold_temp:
            #     mode = 'Heat Pump On'
            # elif self.mode == 'Lower On' and t_lower < self.upper_threshold_temp:
            #     mode = 'Lower On'
            # elif t_lower > self.upper_threshold_temp and t_upper > self.upper_threshold_temp:
            #     mode = 'Off'
            # else:
            #     mode = 'Off'

        # update previous temperature variables
        self.t_upper_prev = t_upper
        self.upper_threshold_temp_prev = self.upper_threshold_temp
        """

        # TODO: Need HPWH control logic validation

        ambient_node = ENVELOPE_NODES[self.air_node] if self.air_node is not None else 'ambient_dry_bulb'
        t_amb = schedule[ambient_node]
        if t_amb < 7.222 or t_amb > 43.333:
            # operate as ERWH
            return super().update_internal_control(schedule)

        t_upper = self.model.get_state('T_' + self.upper_node)
        t_lower = self.model.get_state('T_' + self.lower_node)
        t_control = (3 / 4) * t_upper + (1 / 4) * t_lower

        if not self.hp_only_mode and t_upper < self.upper_threshold_temp - 18.5:
            return 'Upper On'
        elif self.mode != 'Upper On' and t_control < self.upper_threshold_temp - 3.89:
            return 'Heat Pump On'
        elif self.mode == 'Upper On' and t_upper >= self.upper_threshold_temp:
            return 'Off'
        elif t_control >= self.upper_threshold_temp:
            return 'Off'

        # if self.hp_only_mode:
        #     # HP only with no element
        #     if t_control < self.upper_threshold_temp - 3.89 or (
        #             self.mode == 'Heat Pump On' and t_control < self.upper_threshold_temp):
        #         return 'Heat Pump On'
        #     else:
        #         return 'Off'
        # else:
        #     # Standard control logic
        #     if t_upper < self.upper_threshold_temp - 18.5 or (
        #             self.mode == 'Upper On' and t_upper < self.upper_threshold_temp):
        #         return 'Upper On'
        #     elif t_control < self.upper_threshold_temp - 3.89 or (
        #             self.mode == 'Heat Pump On' and t_control < self.upper_threshold_temp):
        #         return 'Heat Pump On'
        #     else:
        #         return 'Off'

    def disperse_hp_heat(self, heat):
        # based on # of nodes, divide delivered heat to various nodes
        if self.model.n_nodes == 2:
            # all heat to lower node
            to_model = {'H_2': heat}  # in W
        elif self.model.n_nodes == 12:
            to_model = {'H_{}'.format(i): heat / 6 for i in range(7, 12)}
            to_model['H_6'] = heat / 12
            to_model['H_12'] = heat / 12
        else:
            raise Exception(
                '{} model not defined for {} with {} nodes'.format(self.name, self.model.name, self.model.n_nodes))
        return to_model

    def update_cop_and_power(self, schedule):
        # TODO: update if HPWH not in Indoor zone
        t_in_wet = schedule['Indoor Wet Bulb']
        t_lower = self.model.get_state('T_' + self.lower_node)
        vector = [1, t_in_wet, t_in_wet ** 2, t_lower, t_lower ** 2, t_lower * t_in_wet]
        self.hp_power = self.hp_power_nominal * sum([x * y for x, y in zip(self.hp_power_coeff, vector)])
        self.cop = self.cop_nominal * sum([x * y for x, y in zip(self.cop_coeff, vector)])

    def calculate_power_and_heat(self, schedule):
        if self.mode == 'Heat Pump On':
            # calculate dynamic capacity and COP
            self.update_cop_and_power(schedule)

            self.delivered_heat = self.hp_power * self.cop
            self.electric_kw = (self.hp_power + self.fan_power) / 1000  # W to kW

            # heat gains to air from HP (sensible and latent) and from fan (sensible only)
            self.sensible_gain = - (self.delivered_heat - self.hp_power) * self.shr_nominal + self.fan_power
            self.latent_gain = - (self.delivered_heat - self.hp_power) * (1 - self.shr_nominal)

            # update water model, add tank losses
            to_model = self.disperse_hp_heat(self.delivered_heat)
            super().update_model(to_model, schedule)

        else:
            self.hp_power = 0
            self.cop = 1 if 'On' in self.mode else 0

            super().calculate_power_and_heat(schedule)
            self.electric_kw += self.parasitic_power / 1000

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if not to_ext:
            if verbosity >= 6:
                results.update({self.end_use + ' Heat Pump Power (kW)': self.hp_power / 1000,
                                self.end_use + ' Heat Pump COP (-)': self.cop})
        return results


class ModulatingWaterHeater(TankWaterHeater):
    """
    Water heater model for low time resolution simulations. Uses the ideal capacity algorithm from IdealHVAC to
    determine the water heater heat delivered to exactly match the tank setpoint temperature. Uses the 1-node water
    tank model, and accounts for tank losses. 
    """
    name = 'Modulating Water Heater'
    model_class = OneNodeWaterModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capacity = 0  # in W
        self.capacity_rated = kwargs['rated input power (W)']  # maximum capacity, in W

        # update initial state to top of deadband
        self.model.x[0] = self.upper_threshold_temp

    def update_external_control(self, schedule, ext_control_args):
        # TODO: use IdealHVAC code to allow duty cycle control
        if ext_control_args.get('Duty Cycle') is not None:
            raise Exception("{} can't handle duty cycle control".format(self.name))

        return super().update_external_control(schedule, ext_control_args)

    def update_internal_control(self, schedule):
        # calculate ideal capacity based on tank model

        # Get water draw inputs and ambient temp
        to_model = self.model.update_water_draw(schedule, self.tempered_draw_temp)
        to_model['T_AMB'] = schedule['Indoor']

        # Solve for desired capacity, subtracting external gains
        self.model.update_inputs(to_model)
        ext_gains = self.model.get_input('H_1')
        h_desired = self.model.solve_for_input('T_1', 'H_1', self.upper_threshold_temp)  # in W
        self.capacity = h_desired - ext_gains

        # Account for efficiency, and only allow heating
        self.capacity = self.capacity / self.efficiency
        self.capacity = np.clip(self.capacity, 0, self.capacity_rated)

        return 'On' if self.capacity > 0 else 'Off'

    def calculate_power_and_heat(self, schedule):
        self.electric_kw = self.capacity / 1000
        self.delivered_heat = self.capacity * self.efficiency  # in W
        self.sensible_gain = self.capacity * (1 - self.efficiency)  # in W

        to_model = {'H_1': self.delivered_heat}
        self.update_model(to_model, schedule)


class GasWaterHeater(TankWaterHeater):
    name = 'Gas Water Heater'
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lower_node = str(self.model.n_nodes)  # bottom node
        self.rated_power = kwargs['rated input power (W)'] / 1000  # in kW
        if kwargs['EF'] < 0.7:
            self.skin_loss_frac = 0.64
        elif kwargs['EF'] < 0.8:
            self.skin_loss_frac = 0.91
        else:
            self.skin_loss_frac = 0.96

    def calculate_power_and_heat(self, schedule):
        if self.mode == 'On':
            self.gas_therms_per_hour = Units.kWh2therms(self.rated_power)  # kW to therms/hour
            self.delivered_heat = self.rated_power * self.efficiency * 1000  # in W
        else:
            # no electric parasitic power
            self.gas_therms_per_hour = 0
            self.delivered_heat = 0
        to_model = {'H_' + self.lower_node: self.delivered_heat}

        # note: no sensible gains from heater (all is vented), tank losses reduced by skin loss frac
        self.sensible_gain = 0
        self.update_model(to_model, schedule)
        self.sensible_gain *= self.skin_loss_frac


class TanklessWaterHeater(TankWaterHeater):
    name = 'Tankless Water Heater'
    model_class = IdealWaterModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Control parameters
        self.tankless_derate = 0.08

    def update_internal_control(self, schedule):
        if schedule.get('water_draw') or schedule.get('tempered_water_draw'):
            return 'On'
        else:
            return 'Off'

    def calculate_power_and_heat(self, schedule):
        heat = self.model.update_water_draw(schedule, set_point_temp=self.tempered_draw_temp)
        if heat:
            # ideal tank: heat delivered is equal and opposite of heat lost from water draw
            self.delivered_heat = -heat.get('H_1', 0)

            # for now, no extra heat gains for tankless water heater
            # heat_to_env = self.delivered_heat * (1 / self.efficiency / (1 - self.tankless_derate) - 1)
            # to_envelope = to_envelope[0] + heat_to_env, to_envelope[1]
            self.sensible_gain = 0

        else:
            # no water draw, no parasitic losses for electric tankless
            self.delivered_heat = 0
            self.sensible_gain = 0

        self.electric_kw = self.delivered_heat / self.efficiency / (1 - self.tankless_derate) / 1000


class GasTanklessWaterHeater(TanklessWaterHeater):
    name = 'Gas Tankless Water Heater'
    is_electric = True  # parasitic power is electric
    is_gas = True

    def calculate_power_and_heat(self, schedule):
        super().calculate_power_and_heat(schedule)

        # gas power in therms/hour
        power_kw = self.delivered_heat / self.efficiency / (1 - self.tankless_derate) / 1000
        self.gas_therms_per_hour = Units.kWh2therms(power_kw)

        if self.mode == 'On':
            self.electric_kw = 65 / 1000  # hardcoded parasitic electric power
        else:
            self.electric_kw = 5 / 1000  # hardcoded electric power
