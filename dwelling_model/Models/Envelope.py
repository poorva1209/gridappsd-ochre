import numpy as np

from dwelling_model import Units, FileIO
from dwelling_model.Models import RCModel, HumidityModel, ModelException

MAIN_NODES = {
    'LIV': 'Indoor',
    'FND': 'Foundation',
    'GAR': 'Garage',
    'ATC': 'Attic'}
EXT_NODES = {'EXT': 'Outdoor',
             'GND': 'Ground'}

# Each boundary has 4 parameters: [Name, Starting node, Ending node, Alternate ending node]
BOUNDARIES = {
    'CL': ('Ceiling', 'ATC', 'LIV', None),
    'FL': ('Floor', 'FND', 'LIV', None),
    'WD': ('Window', 'EXT', 'LIV', None),
    'IW': ('Interior wall', 'LIV', 'LIV', None),
    'EW': ('Exterior wall', 'EXT', 'LIV', None),
    'AW': ('Attached wall', 'GAR', 'LIV', None),
    'GW': ('Garage wall', 'EXT', 'GAR', None),
    'FW': ('Foundation wall', 'GND', 'FND', None),  # below ground
    'CW': ('Crawlspace wall', 'EXT', 'FND', None),  # above ground
    'RF': ('Roof', 'EXT', 'ATC', 'LIV'),
    'RG': ('Roof Gable', 'EXT', 'ATC', 'LIV'),
    'GR': ('Garage roof', 'EXT', 'GAR', None),
    'GF': ('Garage floor', 'GND', 'GAR', None),
    'FF': ('Foundation floor', 'GND', 'FND', 'LIV'),
}
EXT_BOUNDARIES = {key: val for key, val in BOUNDARIES.items() if val[1] == 'EXT'}


class Envelope(RCModel):
    name = 'Envelope'

    def __init__(self, envelope_model_name='Env', **kwargs):
        self.name = envelope_model_name
        self.solar_gain_fraction = {}
        self.liv_resistors = {}

        self.floor_node = None
        self.wall_nodes = None
        self.capacitances = None
        super().__init__(ext_node_names=list(EXT_NODES.keys()), **kwargs)

        # Initialize humidity model
        t_init = self.get_state('T_LIV')
        self.liv_net_sensible_gains = 0
        self.latent_gains = 0
        self.humidity = HumidityModel(t_init, **kwargs)

        # Occupancy parameters
        occupancy_gain = kwargs['gain per occupant (W)'] * kwargs['number of occupants']
        self.occupancy_sensible_gain = (kwargs['occupants convective gainfrac'] +
                                        kwargs['occupants radiant gainfrac']) * occupancy_gain
        self.occupancy_latent_gain = kwargs['occupants latent gainfrac'] * occupancy_gain

        # Infiltration parameters for indoor, basement, and garage zones
        # TODO: add attic infiltration
        self.inf_heat = {node: 0 for node in MAIN_NODES.keys() if 'H_' + node in self.input_names}  # in W
        self.inf_flow = {node: 0 for node in MAIN_NODES.keys() if 'H_' + node in self.input_names}  # in m^3/s
        self.infiltration_parameters = {}
        self.node_volumes = {'LIV': kwargs['building volume (m^3)'],
                             'FND': kwargs.get('basement-volume (m^3)', kwargs.get('crawlspace-volume (m^3)', 0)),
                             # 'GAR': kwargs.get('garage-volume (m^3)', 0),
                             }
        assert kwargs['infiltration-method'] == 'ASHRAE'
        inf_sft = kwargs['inf_f_t'] * (
                kwargs['ws_S_wo'] * (1 - kwargs['inf_Y_i']) + kwargs['inf_S_wflue'] * 1.5 * kwargs['inf_Y_i'])
        self.infiltration_parameters['LIV'] = {'inf_C_i': kwargs['inf_C_i'],
                                               'inf_n_i': kwargs['inf_n_i'],
                                               'inf_stack_coef': kwargs['inf_stack_coef'],
                                               'inf_wind_coef': kwargs['inf_wind_coef'],
                                               'inf_sft': inf_sft}
        if 'FND' in self.inf_heat:
            assert kwargs.get('foundation infiltration-method') == 'ConstantACH'
            self.infiltration_parameters['FND'] = {'ConstantACH': kwargs['infiltration-ACH']}
        if 'GAR' in self.inf_heat:
            assert kwargs.get('garage infiltration-method') == 'ELA'
            self.infiltration_parameters['GAR'] = {
                'ELA': kwargs['garage ELA (cm^2)'],
                'inf_stack_coef': kwargs['garage stack coefficient {(L/s)/(cm^4-K)}'],
                'inf_wind_coef': kwargs['garage wind coefficient {(L/s)/(cm^4-(m/s))}']
            }
        if 'ATC' in self.inf_heat:
            assert kwargs.get('attic infiltration-method') == 'ELA'
            self.infiltration_parameters['ATC'] = {
                'ELA': kwargs['attic ELA (cm^2)'],
                'inf_stack_coef': kwargs['attic stack coefficient {(L/s)/(cm^4-K)}'],
                'inf_wind_coef': kwargs['attic wind coefficient {(L/s)/(cm^4-(m/s))}']
            }

        # Ventilation parameters
        self.ventilation_type = kwargs.get('ventilation_type', 'exhaust')
        self.ventilation_max_flow_rate = kwargs['ventilation_cfm']
        self.ventilation_sens_eff = kwargs.get('erv_sensible_effectiveness', 0)
        self.ventilation_lat_eff = kwargs.get('erv_latent_effectiveness', 0)
        self.vent_flow = 0  # m^3/s, only for LIV node
        self.air_changes = 0  # ACH, only for LIV node

        # Results parameters
        self.temp_deadband = (kwargs.get('heating deadband temperature (C)', 1),
                              kwargs.get('cooling deadband temperature (C)', 1))
        self.unmet_hvac_load = 0

    def load_rc_data(self, **kwargs):
        # Converts RC parameter names to readable format
        rc_params = FileIO.get_rc_params(circuit_name=self.name, **kwargs)

        # save most interior floor node (largest number) and interior wall nodes for solar gains and furniture
        floor_boundary = 'FL' if 'C_FL1' in rc_params else 'FF'
        floor = [floor_boundary + str(i + 1) for i in range(4) if 'C_' + floor_boundary + str(i + 1) in rc_params]
        if floor:
            self.floor_node = floor[-1]
        if 'C_IW1' in rc_params:
            last_wall = ['IW' + str(i + 1) for i in range(4) if 'C_IW' + str(i + 1) in rc_params][-1]
            self.wall_nodes = 'IW1', last_wall

        # Add furniture capacitance to air node, interior wall, and floor
        if 'C_FURN' in rc_params:
            # For now, use empirical fraction based on unit tests
            c_furn = rc_params['C_FURN']
            fraction_to_air = 1
            fraction_to_walls = 0
            fraction_to_floor = 0

            rc_params['C_LIV'] += c_furn * fraction_to_air
            if self.wall_nodes is not None:
                for wall_node in self.wall_nodes:
                    # rc_params['R_IW1'] *= rc_params['C_IW1'] / (rc_params['C_IW1'] + c_furn * fraction_to_walls / 2)
                    rc_params['C_' + wall_node] += c_furn * fraction_to_walls / 2
            if self.floor_node is not None:
                # rc_params['R_FL1'] *= rc_params['C_FL1'] / (rc_params['C_FL1'] + c_furn * fraction_to_walls)
                rc_params['C_' + self.floor_node] += c_furn * fraction_to_floor

            del rc_params['C_FURN']

        # Combine film resistances for exterior boundaries
        for boundary, data in EXT_BOUNDARIES.items():
            r_ext_name = 'R_' + boundary + '_EXT'
            r_int_name = 'R_' + boundary + '1'
            if r_ext_name in rc_params:
                self.solar_gain_fraction[boundary] = \
                    rc_params[r_ext_name] / (rc_params[r_int_name] + rc_params[r_ext_name])
                rc_params[r_int_name] += rc_params[r_ext_name]
                del rc_params[r_ext_name]
            else:
                self.solar_gain_fraction[boundary] = 1

        # parse RC names, get internal node names (internal nodes have a C)
        all_cap = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'C'}
        all_res = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'R'}
        internal_nodes = list(all_cap.keys())

        # save capacitance values
        self.capacitances = all_cap

        # for resistances, convert boundary names to node names
        # e.g. R_CL1 -> R_ATC_CL1,  R_CL2 -> R_CL1_CL2,  R_CL3 -> R_CL2_LIV
        rc_params_new = {name: val for name, val in rc_params.items() if name[0] == 'C'}
        res_used = []
        for boundary, (bd_name, start, end, alt_end) in BOUNDARIES.items():
            # get all relevant R associated with the boundary
            res = {name: val for name, val in all_res.items() if boundary in name}
            if not res:
                continue
            if end not in internal_nodes:
                if alt_end is None:
                    raise ModelException('Error parsing Envelope, {} node must exist for {}'.format(end, bd_name))
                end = alt_end

            # determine order of resistors and capacitors within boundary
            n_res = len(res)
            res_names = [boundary + str(i + 1) for i in range(n_res)]
            r_values = [all_res[name] for name in res_names]
            res_used.extend(res_names)
            if start == end and n_res == 1:
                # force resistor to connect start to "IW1"
                node_list = [boundary + str(1), end]
            else:
                node_list = [start] + [boundary + str(i + 1) for i in range(n_res - 1)] + [end]

            # update R names
            for i, r_value in enumerate(r_values):
                res_name = '_'.join(['R', node_list[i], node_list[i + 1]])
                if res_name in rc_params_new:
                    rc_params_new[res_name] = self.par(rc_params_new[res_name], r_value)
                else:
                    rc_params_new[res_name] = r_value

            # save resistors connected to living space
            if end == 'LIV':
                self.liv_resistors[bd_name] = (node_list[-2], r_values[-1])
                if start == 'LIV' and n_res > 1:
                    self.liv_resistors[bd_name + ' 2'] = (node_list[1], r_values[0])

        if len(res_used) < len(all_res):
            unused = [r for r in all_res.keys() if r not in res_used]
            raise ModelException('Some resistor values not used in model: {}'.format(unused))

        return rc_params_new

    def load_initial_state(self, initial_temp_setpoint=None, **kwargs):
        # Sets all temperatures to the steady state value based on initial conditions
        # Adds random temperature if exact setpoint is not set
        # Note: initialization will update the initial state to more typical values
        outdoor_temp = kwargs['initial_schedule']['ambient_dry_bulb']
        ground_temp = kwargs['initial_schedule']['ground_temperature']



        # Indoor initial condition depends on ambient temperature - use heating/cooling setpoint when </> 15 deg C
        if initial_temp_setpoint is None:
            # select heating/cooling setpoint based on starting outdoor temperature
            if outdoor_temp > 12:
                deadband = kwargs.get('cooling deadband temperature (C)', 1)
                random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
                indoor_temp = kwargs['initial_schedule']['cooling_setpoint'] + random_delta
            else:
                deadband = kwargs.get('heating deadband temperature (C)', 1)
                random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
                indoor_temp = kwargs['initial_schedule']['heating_setpoint'] + random_delta
        elif isinstance(initial_temp_setpoint, str):
            assert initial_temp_setpoint in ['heating', 'cooling']
            deadband = kwargs.get(initial_temp_setpoint + ' deadband temperature (C)', 1)
            random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
            indoor_temp = kwargs['initial_schedule'][initial_temp_setpoint + '_setpoint'] + random_delta
        elif isinstance(initial_temp_setpoint, (int, float)):
            indoor_temp = initial_temp_setpoint
        else:
            raise ModelException('Unknown initial temperature setpoint: {}'.format(initial_temp_setpoint))

        # Update continuous time matrices to swap T_LIV from state to input
        x_idx = self.state_names.index('T_LIV')
        keep_states = [i for i in range(len(self.state_names)) if i != x_idx]
        keep_inputs = [self.input_names.index('T_EXT'), self.input_names.index('T_GND')]
        A = self.A_c[keep_states, :][:, keep_states]
        B = np.hstack((self.A_c[keep_states, x_idx:x_idx + 1], self.B_c[keep_states, :][:, keep_inputs]))
        u = np.array([indoor_temp, outdoor_temp, ground_temp])

        # Calculate steady state values (effectively interpolates from the input temperatures)
        x = - np.linalg.inv(A).dot(B).dot(u)
        x = np.insert(x, x_idx, indoor_temp)
        return x

    def remove_unused_inputs(self, unused_inputs=None, **kwargs):
        # remove heat injection inputs that aren't into main nodes or nodes with a solar gain injection
        unused_inputs = ['H_{}{}'.format(bd, i) for i in range(2, 4) for bd in BOUNDARIES]
        unused_inputs.extend(['H_{}1'.format(bd) for bd in BOUNDARIES if bd not in EXT_BOUNDARIES])

        if self.floor_node is not None:
            unused_inputs.pop(unused_inputs.index('H_' + self.floor_node))

        # find first and last interior wall nodes for solar gains through window
        if self.wall_nodes is not None:
            unused_inputs.pop(unused_inputs.index('H_' + self.wall_nodes[0]))
            if self.wall_nodes[1] != self.wall_nodes[0]:
                unused_inputs.pop(unused_inputs.index('H_' + self.wall_nodes[1]))

        super().remove_unused_inputs(unused_inputs=unused_inputs, **kwargs)

    def update_infiltration(self, t_ext, wind_speed, ventilation_cfm):
        # calculate infiltration for all main nodes, add ventilation for Indoor node only

        # calculate Indoor infiltration flow (m^3/s)
        delta_t = t_ext - self.get_state('T_LIV')
        params = self.infiltration_parameters['LIV']

        inf_c = Units.cfm2m3_s(params['inf_C_i']) / Units.inH2O2Pa(1.0) ** params['inf_n_i']
        inf_Cs = params['inf_stack_coef'] * Units.inH2O_R2Pa_K(1.0) ** params['inf_n_i']
        inf_Cw = params['inf_wind_coef'] * Units.inH2O_mph2Pas2_m2(1.0) ** params['inf_n_i']
        inf_flow_temp = inf_c * inf_Cs * abs(delta_t) ** params['inf_n_i']
        inf_flow_wind = inf_c * inf_Cw * (params['inf_sft'] * wind_speed) ** (2 * params['inf_n_i'])
        self.inf_flow['LIV'] = (inf_flow_temp ** 2 + inf_flow_wind ** 2) ** 0.5

        # calculate Indoor ventilation flow (m^3/s)
        self.vent_flow = Units.cfm2m3_s(ventilation_cfm)

        # combine and calculate ACH, Indoor node only
        if self.ventilation_type == 'balanced':
            # Add balanced + unbalanced in quadrature, but both inf and balanced are same term
            total_flow = self.inf_flow['LIV'] + self.vent_flow
        else:
            total_flow = (self.inf_flow['LIV'] ** 2 + self.vent_flow ** 2) ** 0.5
        self.air_changes = total_flow / self.node_volumes['LIV'] * 3600  # Air changes per hour

        # calculate indoor node sensible heat gain
        cp_air = 1.005  # kJ/kg/K
        # TODO: For now we're only handling HRV, not ERV. Probably just need to sum sensible + latent, but Jeff to double check E+
        ventilation_erv_eff = self.ventilation_sens_eff + self.ventilation_lat_eff
        if ventilation_erv_eff > 0:
            # if effectiveness is 70%, 70% of heat is recovered so HX to space is (1-.7) = 30%
            self.inf_heat['LIV'] = self.inf_flow['LIV'] * delta_t * self.humidity.indoor_density * cp_air * 1000 + \
                                   self.vent_flow * delta_t * self.humidity.indoor_density * cp_air * 1000 * (
                                           1 - ventilation_erv_eff)
        else:
            self.inf_heat['LIV'] = total_flow * delta_t * self.humidity.indoor_density * cp_air * 1000  # in W

        # TODO: add latent gains for indoor node infiltration

        # calculate infiltration heat gain from foundation
        # TODO: clean up, remove copy/pasted code; allow for multiple types of infiltration methods, combine with above
        if 'FND' in self.inf_heat:
            delta_t = t_ext - self.get_state('T_FND')
            self.inf_flow['FND'] = self.infiltration_parameters['FND']['ConstantACH'] * self.node_volumes['FND'] / 3600
            heat_flow = self.inf_flow['FND'] * self.humidity.indoor_density * cp_air * 1000  # in W / K
            if heat_flow * self.time_res.total_seconds() > self.capacitances['FND']:
                print('WARNING: Limiting Foundation heat gains due to large flow rate:', self.inf_flow['FND'])
                heat_flow = self.capacitances['FND'] / self.time_res.total_seconds()
            self.inf_heat['FND'] = heat_flow * delta_t  # in W
        if 'GAR' in self.inf_heat:
            # see https://bigladdersoftware.com/epx/docs/8-6/input-output-reference/group-airflow.html
            #  - zoneinfiltrationeffectiveleakagearea
            delta_t = t_ext - self.get_state('T_GAR')
            f = 1
            params = self.infiltration_parameters['GAR']
            self.inf_flow['GAR'] = f * params['ELA'] / 1000 * (
                    params['inf_stack_coef'] * abs(delta_t) + params['inf_wind_coef'] * wind_speed ** 2) ** 0.5
            heat_flow = self.inf_flow['GAR'] * self.humidity.indoor_density * cp_air * 1000  # in W / K
            if heat_flow * self.time_res.total_seconds() > self.capacitances['GAR']:
                # print('WARNING: Limiting Garage heat gains due to large flow rate:', self.inf_flow['GAR'])
                heat_flow = self.capacitances['GAR'] / self.time_res.total_seconds()
            self.inf_heat['GAR'] = heat_flow * delta_t  # in W
        if 'ATC' in self.inf_heat:
            # same method as garage
            delta_t = t_ext - self.get_state('T_ATC')
            f = 1
            params = self.infiltration_parameters['ATC']
            self.inf_flow['ATC'] = f * params['ELA'] / 1000 * (
                    params['inf_stack_coef'] * abs(delta_t) + params['inf_wind_coef'] * wind_speed ** 2) ** 0.5
            heat_flow = self.inf_flow['ATC'] * self.humidity.indoor_density * cp_air * 1000  # in W / K
            if heat_flow * self.time_res.total_seconds() > self.capacitances['ATC']:
                # TODO: uncomment below when attic capacitance is increased
                # print('WARNING: Limiting Attic heat gains due to large flow rate:', self.inf_flow['ATC'])
                heat_flow = self.capacitances['ATC'] / self.time_res.total_seconds()
            self.inf_heat['ATC'] = heat_flow * delta_t  # in W

        # FUTURE: add latent gains for other nodes

    def get_model_inputs(self, to_model, schedule):
        # Add solar gains to exterior boundaries (e.g. roof, windows, garage walls...)
        for boundary, data in EXT_BOUNDARIES.items():
            solar_gain = 'H_{}1'.format(boundary)
            if solar_gain not in schedule:
                continue

            if boundary == 'WD':
                # and window solar gains to a mix of air node, interior walls, and floor
                fraction_to_air = 0.0
                fraction_to_walls = 0.9
                fraction_to_floor = 0.1

                to_model['H_LIV'] += schedule[solar_gain] * fraction_to_air

                if self.wall_nodes:
                    # inject into both sides of interior walls
                    for wall_node in self.wall_nodes:
                        to_model['H_' + wall_node] += schedule[solar_gain] * fraction_to_walls / 2

                if self.floor_node:
                    to_model['H_' + self.floor_node] += schedule[solar_gain] * fraction_to_floor

            else:
                to_model[solar_gain] += schedule[solar_gain] * self.solar_gain_fraction[boundary]

        # Add occupancy heat and latent gains
        to_model['H_LIV'] += schedule['Occupancy'] * self.occupancy_sensible_gain  # in W
        to_model['H_LIV_latent'] += schedule['Occupancy'] * self.occupancy_latent_gain

        # Add infiltration and ventilation
        vent_cfm = schedule['ventilation_rate'] * self.ventilation_max_flow_rate
        self.update_infiltration(schedule['ambient_dry_bulb'], schedule['wind_speed'], vent_cfm)
        for node in self.inf_heat:
            to_model['H_' + node] += self.inf_heat[node]
            # TODO: add latent gains

        # Add external temperatures to inputs
        to_model['T_EXT'] = schedule['ambient_dry_bulb']
        if 'T_GND' in self.input_names:
            to_model['T_GND'] = schedule['ground_temperature']

        return to_model

    def update(self, to_model, schedule=None):
        to_model = self.get_model_inputs(to_model, schedule)
        self.liv_net_sensible_gains = to_model.get('H_LIV', 0)
        self.latent_gains = to_model.pop('H_LIV_latent')

        # Run RC Model update
        states = super().update(to_model)
        t_liv = states['T_LIV']

        # check that states are within reasonable range
        if any([(temp < -30) or (temp > 80) for temp in states.values()]):
            print('WARNING: Extreme envelope temperatures: {}'.format(states))
        if any([(temp < -40) or (temp > 90) for temp in states.values()]):
            raise ModelException('Envelope temperatures are outside acceptable range: {}'.format(states))

        # Run humidity update
        # NOTE: Only incorporating latent gains in LIV node
        self.humidity.update_humidity(t_liv, self.air_changes, schedule['ambient_dry_bulb'],
                                      schedule['ambient_humidity'], schedule['ambient_pressure'], self.latent_gains)
        if t_liv < self.humidity.indoor_wet_bulb - 0.1:
            print('Warning: Wet bulb temp ({}), greater than dry bulb temp ({})'.format(
                self.humidity.indoor_wet_bulb, t_liv))
            self.humidity.indoor_wet_bulb = t_liv

        # Calculate unmet thermal loads - negative=below deadband, positive=above deadband
        t_low = schedule['heating_setpoint'] - self.temp_deadband[0] / 2
        t_high = schedule['cooling_setpoint'] + self.temp_deadband[1] / 2
        self.unmet_hvac_load = t_liv - t_low if t_liv < t_low else max(t_liv - t_high, 0)

    def get_main_states(self):
        # get temperatures from main nodes, and living space humidity
        out = {name: self.get_state('T_' + key) for key, name in MAIN_NODES.items() if 'T_' + key in self.state_names}
        out['Indoor Wet Bulb'] = self.humidity.indoor_wet_bulb
        return out

    def generate_results(self, verbosity, to_ext=False):
        main_temps = self.get_main_states()

        if to_ext:
            # for now, use different format for external controller
            to_ext_control = {'T {}'.format(loc): val for loc, val in main_temps.items()}
            return to_ext_control
        else:
            results = {}
            if verbosity >= 1:
                results.update({'Temperature - {} (C)'.format(loc): val for loc, val in main_temps.items()})
                results.update({'Temperature - {} (C)'.format(name): self.get_input('T_' + node) for node, name in
                                EXT_NODES.items() if 'T_' + node in self.input_names})
                results['Unmet HVAC Load (C)'] = self.unmet_hvac_load
            if verbosity >= 4:
                results.update({
                    'Relative Humidity - Indoor (-)': self.humidity.indoor_rh,
                    'Humidity Ratio - Indoor (-)': self.humidity.indoor_w,
                    'Indoor Net Sensible Heat Gain (W)': self.liv_net_sensible_gains,
                    'Indoor Net Latent Heat Gain (W)': self.latent_gains,

                    'Air Changes per Hour (1/hour)': self.air_changes,
                    'Indoor Ventilation Flow Rate (m^3/s)': self.vent_flow,
                })
                results.update({
                    MAIN_NODES[node] + ' Infiltration Flow Rate (m^3/s)': val for node, val in self.inf_flow.items()
                })
                results.update({
                    MAIN_NODES[node] + ' Infiltration Heat Gain (W)': val for node, val in self.inf_heat.items()
                })

                # add heat injections into the living space (pos=heat injected)
                t_indoor = self.get_state('T_LIV')
                for boundary, (node, r) in self.liv_resistors.items():
                    t_node = self.get_state('T_' + node) if 'T_' + node in self.state_names else self.get_input(
                        'T_' + node)
                    results['Convection from {} (W)'.format(boundary)] = (t_node - t_indoor) / r

            if verbosity >= 8:
                results.update({**self.get_states(), **self.get_inputs()})

            return results
