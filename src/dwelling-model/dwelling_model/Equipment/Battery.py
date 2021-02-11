# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:28:35 2019

@author: rchintal, xjin, mblonsky
"""

import numpy as np
import datetime as dt

from . import Equipment

CONTROL_TYPES = ['Schedule', 'Self-Consumption', 'Off']


class Battery(Equipment):
    name = 'Battery'
    end_use = 'Battery'

    def __init__(self, control_type='Off', parameter_file='default_parameters.csv', **kwargs):

        super().__init__(parameter_file=parameter_file, **kwargs)

        # Nominal capacity parameters for energy (in kWH) and power (in kW)
        self.capacity_kwh = self.parameters['capacity_kwh']
        self.capacity_kw = self.parameters['capacity_kw']

        # power parameters, all in kW
        self.power_setpoint = 0  # setpoint from controller, AC side (after losses)

        # Electrical parameters
        self.soc = self.parameters['soc_init']  # Initial State of Charge
        self.soc_max = self.parameters['soc_max']
        self.soc_min = self.parameters['soc_min']
        self.eta_charge = self.parameters['eta_charge']  # Charging current efficiency
        self.eta_discharge = self.parameters['eta_discharge']  # Discharge current efficiency
        self.discharge_pct = self.parameters['discharge_pct']  # Self-discharge rate (% per day)
        self.n_series = self.parameters['n_series']  # Number of cells in series
        self.r_cell = self.parameters['r_cell']  # Cell resistance in ohms (for voltage calculation only)
        self.initial_voltage = self.parameters['initial_voltage']  # Initial open circuit voltage

        self.capacity_ah = (self.capacity_kwh / self.initial_voltage) * 1000  # Battery capacity in Ah
        self.voltage_terminal = self.initial_voltage
        self.charge = self.soc * self.capacity_ah  # in Ah

        # Voltage curve constants (hard-coded
        self.K = 0.00876  # Battery constant
        self.a = 0.468  # Battery constant
        self.B = 3.5294  # Battery constant

        # controller parameters
        if control_type not in CONTROL_TYPES:
            raise Exception('Unknown {} control type: {}'.format(self.name, control_type))
        self.internal_control_type = control_type

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        #   - Note: still subject to SOC limits and charge/discharge limits
        # - Control Type: Sets the control type ('Schedule', 'Self-Consumption', 'Off')
        # - Parameters: Update control parameters, including:
        #   - Schedule charge/discharge start times
        #   - Schedule charge/discharge powers
        #   - Self-Consumption charge/discharge offsets, in kW
        #   - Self-Consumption charge type (from any solar or from net power)

        if 'Parameters' in ext_control_args:
            self.parameters.update(ext_control_args['Parameters'])

        if 'Controller' in ext_control_args:
            control_type = ext_control_args['Controller']
            if control_type in CONTROL_TYPES:
                self.internal_control_type = control_type
            else:
                print('Unknown control type for {} ({}). Keeping previous control type'.format(self.name, control_type))

        # set power directly from setpoint
        if 'P Setpoint' in ext_control_args:
            self.power_setpoint = ext_control_args['P Setpoint']
            return 'On' if self.power_setpoint != 0 else 'Off'

        return self.update_internal_control(schedule)

    def update_internal_control(self, schedule):
        # Set power setpoint based on internal control type

        if self.internal_control_type == 'Schedule':
            # Charges or discharges at given power and given time of day
            time = self.current_time.time()
            if time == dt.time(hour=int(self.parameters['charge_start_hour'])):
                self.power_setpoint = self.parameters['charge_power']
            if time == dt.time(hour=int(self.parameters['discharge_start_hour'])):
                self.power_setpoint = - self.parameters['discharge_power']

        elif self.internal_control_type == 'Self-Consumption':
            net_power = schedule.get('net_power', 0)
            # Discharges based on net load only
            discharge_limit = min(- net_power + self.parameters.get('discharge_limit_offset'), 0)

            # Option to charges based on PV or net load
            if self.parameters['charge_any_solar']:
                pv_power = schedule.get('pv_power', 0)
                charge_limit = max(- pv_power - self.parameters.get('charge_limit_offset'), 0)
            else:
                charge_limit = max(- net_power - self.parameters.get('charge_limit_offset'), 0)

            if charge_limit != 0 and discharge_limit != 0:
                # Rare case - this will limit total charging/discharging if both are allowed
                self.power_setpoint = (charge_limit + discharge_limit) / 2
            elif charge_limit != 0:
                self.power_setpoint = charge_limit
            else:
                self.power_setpoint = discharge_limit

        elif self.internal_control_type == 'Off':
            self.power_setpoint = 0

        # Update setpoint if SOC limits are reached
        if self.power_setpoint > 0 and self.soc >= self.soc_max:
            self.power_setpoint = 0
        if self.power_setpoint < 0 and self.soc <= self.soc_min:
            self.power_setpoint = 0

        return 'On' if self.power_setpoint != 0 else 'Off'

    def calculate_power_and_heat(self, schedule):
        if self.mode == 'Off':
            self.electric_kw = 0
            self.sensible_gain = 0
            return

        # force ac power within kw capacity and SOC limits
        hours = self.time_res.total_seconds() / 3600
        max_charge = min((self.soc_max - self.soc) * self.capacity_kwh / hours / self.eta_charge, self.capacity_kw)
        max_discharge = min((self.soc - self.soc_min) * self.capacity_kwh / hours * self.eta_discharge,
                            self.capacity_kw)
        ac_power = np.clip(self.power_setpoint, -max_discharge, max_charge)
        self.electric_kw = ac_power

        # calculate internal battery power and power loss
        if ac_power >= 0:  # charging
            dc_power = ac_power * self.eta_charge
        else:
            dc_power = ac_power / self.eta_discharge
        self.sensible_gain = (ac_power - dc_power) * 1000  # power losses, in W
        assert self.sensible_gain >= 0

        # update SOC and charge, check with upper and lower bound of usable SOC
        self_discharge = self.discharge_pct / 100 * hours / 24
        self.soc += dc_power * hours / self.capacity_kwh - self_discharge
        self.charge = self.soc * self.capacity_ah
        assert self.soc_max + 0.001 >= self.soc >= self.soc_min - 0.001  # small computational errors possible

    def update_voltage(self, power):
        # Note: not currently used, not validated
        current_batt = power / self.voltage_terminal
        current_cell = current_batt / self.n_series
        V_cell = (self.initial_voltage / self.n_series) + self.r_cell * current_cell - self.K * (
                self.capacity_ah / self.charge) + self.a * np.exp(
            -self.B * current_cell * self.time_res.total_seconds() / 3600)
        self.voltage_terminal = V_cell * self.n_series

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            if self.mode == 'On':
                mode = 0 if self.power_setpoint >= 0 else 1
            else:
                mode = 2
            return {self.name: {'SOC': self.soc,
                                'Mode': mode}}
        else:
            if verbosity >= 3:
                results.update({self.name + ' SOC (-)': self.soc})
            if verbosity >= 6:
                results.update({self.name + ' Setpoint (kW)': self.power_setpoint})
        return results

    def get_kwh_remaining(self, discharge=True, include_efficiency=True):
        # returns the remaining SOC, in units of kWh. Option for remaining charging/discharging
        # if include_efficiency: return kWh AC (incorporating efficiency). Otherwise, return kWh DC
        if discharge:
            kwh_rem = (self.soc - self.soc_min) * self.capacity_kwh
            if include_efficiency:
                kwh_rem *= self.eta_discharge
        else:
            kwh_rem = (self.soc_max - self.soc) * self.capacity_kwh
            if include_efficiency:
                kwh_rem /= self.eta_charge

        return kwh_rem
