import numpy as np

from dwelling_model.Models import RCModel, ModelException
from dwelling_model import Units

# Water Constants
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L


class StratifiedWaterModel(RCModel):
    """
    Stratified Water Tank RC Thermal Model

    - Partitions a water tank into n nodes (12 by default).
    - Nodes can have different volumes, but are equal volume by default.
    - Node 1 is at the top of the tank (at outlet).
    - State names are [T1, T2, ...], length n
    - Input names are [T_AMB, Q1, Q2, ...], length n+1
    - The model can accept 2 additional inputs for water draw:
      - draw: volume of water to deliver
      - draw_tempered: volume to deliver at setpoint temperature.
      If tank temperature is higher than setpoint, the model assumes mixing with water mains.
    - The model considers the following effects on temperature at each time step:
      - Internal (node-to-node) conduction
      - External (node-to-ambient) conduction
      - Heat injections due to water heater
      - Heat injections due to water draw (calculated before the state-space update)
      - Heat transfer due to inversion mixing (assumes fast mixing, calculated after the state-space update)
    - At each time step, the model calculates:
      - The internal states (temperatures)
      - The heat delivered to the load (relative to mains temperature)
      - The heat lost to ambient air
    """
    name = 'Stratified Water Model'

    def __init__(self, water_nodes=12, vol_fractions=None, **kwargs):
        if vol_fractions is None:
            self.n_nodes = water_nodes
            self.vol_fractions = np.ones(self.n_nodes) / self.n_nodes
        else:
            self.n_nodes = len(vol_fractions)
            self.vol_fractions = np.array(vol_fractions) / sum(vol_fractions)

        self.volume = None  # in L
        super().__init__(**kwargs)

        # key variables for results
        self.draw_total = 0  # in L
        self.h_delivered = 0  # heat delivered in outlet water, in W
        self.h_injections = 0  # heat from water heater, in W
        self.h_loss = 0  # conduction heat loss from tank, in W
        self.h_unmet_shower = 0  # unmet load from lower outlet temperature, showers only, in W
        self.mains_temp = 0  # water mains temperature, in C
        self.outlet_temp = 0  # temperature of outlet water, in C

    def load_rc_data(self, **kwargs):
        # Get properties from input file
        r = kwargs['tank radius (m)'] if 'tank radius (m)' in kwargs else kwargs['Tank Radius (m)']  # in m
        h = kwargs['tank height (m)'] if 'tank height (m)' in kwargs else kwargs['Tank Height (m)']  # in m
        top_area = np.pi * r ** 2
        self.volume = top_area * h * 1000  # in L

        if 'Heat Transfer Coefficient (W/m^2/K)' in kwargs:
            u = kwargs['Heat Transfer Coefficient (W/m^2/K)']
        elif 'UA (W/K)' in kwargs:
            ua = kwargs['UA (W/K)']
            total_area = 2 * top_area + 2 * np.pi * r
            u = ua / total_area
        else:
            raise ModelException('Missing heat transfer coefficient (UA) for {}'.format(self.name))

        # calculate general RC parameters for whole tank
        c_water_tot = self.volume * water_cp * water_density_liters * 1000  # Heat capacity of water (J/K)
        r_int = (h / self.n_nodes) / water_conductivity / top_area  # R between nodes (K/W)
        r_side_tot = 1 / u / (2 * np.pi * r * h)  # R from side of tank (K/W)
        r_top = 1 / u / top_area  # R from top/bottom of tank (K/W)

        # Capacitance per node
        rc_params = {'C_' + str(i + 1): c_water_tot * frac for i, frac in enumerate(self.vol_fractions)}

        # Resistance to exterior from side, top, and bottom
        rc_params.update({'R_{}_AMB'.format(i + 1): r_side_tot / frac for i, frac in enumerate(self.vol_fractions)})
        rc_params['R_1_AMB'] = self.par(rc_params['R_1_AMB'], r_top)
        rc_params['R_{}_AMB'.format(self.n_nodes)] = self.par(rc_params['R_{}_AMB'.format(self.n_nodes)], r_top)

        # Resistance between nodes
        if self.n_nodes > 1:
            rc_params.update({'R_{}_{}'.format(i + 1, i + 2): r_int for i in range(self.n_nodes - 1)})

        return rc_params

    def load_initial_state(self, **kwargs):
        t_max = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        t_db = Units.deltaF2C(kwargs.get('deadband temperature (F)', 10))
        # temp = t_max - np.random.rand(1) * t_db

        # set initial temperature close to top of deadband
        temp = t_max - t_db / 10
        x0 = np.ones(len(self.state_names)) * temp
        return np.array(x0, dtype=float)

    def update_water_draw(self, schedule, set_point_temp=None):
        self.mains_temp = schedule.get('mains_temperature')
        self.outlet_temp = self.get_state('T_1')

        draw = schedule.get('water_draw', 0)
        draw_tempered = schedule.get('tempered_water_draw', 0)
        if not (draw or draw_tempered):
            # No water draw
            self.draw_total = 0
            self.h_delivered = 0
            self.h_unmet_shower = 0
            return {}

        if self.mains_temp is None:
            raise ModelException('Mains temperature required when water draw exists')
        if set_point_temp is None:
            raise ModelException('Set point temperature required when using tempered draw volume')

        # calculate draw volume from tempered draw volume
        # for tempered draw, assume outlet temperature == T1, slightly off if the water draw is very large
        if draw_tempered:
            if self.outlet_temp <= set_point_temp:
                draw += draw_tempered
            else:
                vol_ratio = (set_point_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
                draw += draw_tempered * vol_ratio

        self.draw_total = draw
        t_s = self.time_res.total_seconds()
        draw_liters = self.draw_total * t_s / 60
        draw_fraction = draw_liters / self.volume  # unitless

        if self.n_nodes == 2 and draw_fraction < self.vol_fractions[1]:
            # Use empirical factor for determining water flow by node
            flow_fraction = 0.95  # Totally empirical factor based on detailed lab validation
            if draw_fraction > self.vol_fractions[0]:
                # outlet temp is volume-weighted average of lower and upper temps
                self.outlet_temp = (self.x[0] * self.vol_fractions[0] + self.x[1] * (
                        draw_fraction - self.vol_fractions[0])) / draw_fraction
            q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J

            # q_to_mains_upper = self.volume * self.vol_fractions[0] * water_c * (self.x[0] - self.mains_temp)
            q_to_mains_lower = self.volume * self.vol_fractions[1] * water_c * (self.x[1] - self.mains_temp)
            if q_delivered * flow_fraction > q_to_mains_lower:
                # If you'd fully cool the bottom node to mains, set bottom node to mains and cool top node
                q_nodes = [q_to_mains_lower - q_delivered, -q_to_mains_lower]
            else:
                q_nodes = [-q_delivered * (1 - flow_fraction), -q_delivered * flow_fraction]

        else:
            if draw_fraction < min(self.vol_fractions):
                # water draw is smaller than all node volumes
                q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J
                # all volume transfers are from the node directly below
                q_nodes = draw_liters * water_c * np.diff(self.x, append=self.mains_temp)  # in J
            else:
                # calculate volume transfers to/from each node, including q_delivered
                vols_pre = np.append(self.vol_fractions, draw_fraction).cumsum()
                vols_post = np.insert(self.vol_fractions, 0, draw_fraction).cumsum()
                temps = np.append(self.x, self.mains_temp)

                # update outlet temp as a weighted average of temps, by volume
                vols_delivered = np.diff(vols_pre.clip(max=draw_fraction), prepend=0)
                self.outlet_temp = np.dot(temps, vols_delivered) / draw_fraction
                q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J

                # calculate heat in/out of each node (in J)
                q_nodes = []
                for i in range(self.n_nodes):
                    t_start = self.x[i]
                    vol_frac = self.vol_fractions[i]
                    vols_delivered = np.diff(vols_pre.clip(min=vols_post[i], max=vols_post[i + 1]),
                                             prepend=vols_post[i])
                    t_end = np.dot(temps, vols_delivered) / vol_frac
                    q_nodes.append(vol_frac * self.volume * water_cp * water_density_liters * (t_end - t_start) * 1000)

        # convert heat transfer from J to W
        self.h_delivered = q_delivered / t_s  # Converting J to W
        to_model = {'H_' + str(i + 1): q / t_s for i, q in enumerate(q_nodes)}

        # calculate unmet loads, in W
        shower_draw = schedule.get('shower_draw', 0) / 60  # in L/sec
        self.h_unmet_shower = max(shower_draw * water_c * (set_point_temp - self.outlet_temp), 0)

        return to_model

    def run_inversion_mixing_rule(self):
        # Inversion Mixing Rule
        # See https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v9.1.0/EngineeringReference.pdf
        #     p. 1528
        # Starting from the top, check for mixing at each node
        for node in range(self.n_nodes - 1):
            current_temp = self.x[node]

            # new temp is the max of any possible mixings
            heats = self.x * self.vol_fractions  # note: excluding c_p and volume factors
            # TODO: this is slow, any way to speed up?
            temps = [heats[node: node + i + 1].sum() / self.vol_fractions[node: node + i + 1].sum() for i in
                     range(self.n_nodes - node)]
            new_temp = max(temps)

            # Allow inversion mixing if a significant difference in temperature exists
            if new_temp > current_temp + 0.001:  # small computational errors are possible
                # print('Inversion mixing occuring at node {}. Temperature raises from {} to {}'.format(
                #     node + 1, self.x[node], new_temp))

                # calculate heat transfer, update temperatures of current node and node below
                q = (new_temp - current_temp) * self.vol_fractions[node]
                self.x[node] = new_temp
                self.x[node + 1] -= q / self.vol_fractions[node + 1]

            elif new_temp < current_temp - 0.001:  # small computational errors are possible:
                msg = 'Error in inversion mixing algorithm. ' \
                      'New temperature ({}) less than previous ({}) at node {}.'
                raise ModelException(msg.format(new_temp, self.x[node], node + 1))

    def update(self, to_model, setpoint_temp=None, schedule=None):
        self.h_injections = sum(to_model.values())

        # update heat injections from water draw
        draw_to_model = self.update_water_draw(schedule, setpoint_temp)
        for key, val in draw_to_model.items():
            if key in to_model:
                to_model[key] += val
            else:
                to_model[key] = val

        # update water tank model
        # TODO: update with WH location
        to_model['T_AMB'] = schedule['Indoor']
        q_initial = np.dot(self.x, self.vol_fractions) * self.volume * water_cp * water_density_liters * 1000  # in J
        super().update(to_model)
        q_final = np.dot(self.x, self.vol_fractions) * self.volume * water_cp * water_density_liters * 1000  # in J
        h_change = (q_final - q_initial) / self.time_res.total_seconds()

        # calculate heat loss, in W
        self.h_loss = self.h_injections - h_change - self.h_delivered
        if abs(self.h_loss) > 1000:
            raise ModelException('Error in calculating heat loss for {} model'.format(self.name))

        # If any temperatures are inverted, run inversion mixing algorithm
        if any(np.diff(self.x) > 0.1):
            self.run_inversion_mixing_rule()

        # check final heat to ensure no losses from mixing
        heat_check = np.dot(self.x, self.vol_fractions) * self.volume * water_cp * water_density_liters * 1000  # in J
        if not abs(q_final - heat_check) < 1:
            raise ModelException(
                'Large error in water heater inversion mixing algorithm.'
                'Total initial heat ({} J) differs from final heat ({} J).'.format(q_final, heat_check))

        # check that states are within reasonable range
        # Note: default max temp on water heater model is 60C (140F). Temps may exceed that slightly
        if max(self.x) > 65 or min(self.x) < self.mains_temp - 5:
            raise ModelException('Water temperatures are outside acceptable range: {}'.format(self.x))

        # return the heat loss for envelope model
        return self.h_loss

    def generate_results(self, verbosity, to_ext=False):
        if to_ext:
            return {}
        else:
            results = {}
            if verbosity >= 3:
                results.update({'Hot Water Delivered (L/min)': self.draw_total,
                                'Hot Water Outlet Temperature (C)': self.outlet_temp,
                                'Hot Water Delivered (kW)': self.h_delivered / 1000,
                                'Hot Water Unmet Demand, Showers (kW)': self.h_unmet_shower / 1000,
                                })
            if verbosity >= 6:
                results.update({'Hot Water Heat Injected (kW)': self.h_injections / 1000,
                                'Hot Water Heat Loss (kW)': self.h_loss / 1000,
                                'Hot Water Average Temperature (C)': sum(self.x * self.vol_fractions),
                                'Hot Water Maximum Temperature (C)': max(self.x),
                                'Hot Water Minimum Temperature (C)': min(self.x),
                                'Hot Water Mains Temperature (C)': self.mains_temp,
                                })
            if verbosity >= 9:
                results.update(self.get_states())
                results.update(self.get_inputs())
            return results


class OneNodeWaterModel(StratifiedWaterModel):
    """
    1-node Water Tank Model
    """
    name = 'One-Node Water Model'

    def __init__(self, **kwargs):
        super().__init__(water_nodes=1, **kwargs)


class TwoNodeWaterModel(StratifiedWaterModel):
    """
    2-node Water Tank Model

    - Partitions tank into 2 nodes
    - Top node is 1/3 of volume, Bottom node is 2/3
    """
    name = 'Two-Node Water Model'

    def __init__(self, **kwargs):
        super().__init__(water_nodes=2, vol_fractions=[1 / 3, 2 / 3], **kwargs)


class IdealWaterModel(OneNodeWaterModel):
    """
    Ideal water tank with near-perfect insulation. Used for TanklessWaterHeater. Modeled as 1-node tank.
    """
    name = 'Ideal Water Model'

    def load_rc_data(self, **kwargs):
        # ignore RC parameters from the properties file
        self.volume = 1000
        return {'R_1_AMB': 1e6,
                'C_1': self.volume * water_cp * water_density_liters * 1000}

    def load_initial_state(self, **kwargs):
        # set temperature to upper threshold
        t_max = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        x0 = np.ones(len(self.state_names)) * t_max
        return np.array(x0, dtype=float)
