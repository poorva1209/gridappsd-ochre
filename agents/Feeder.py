from constants import *
from agents import Agent

from opendss_wrapper import OpenDSS


class Feeder(Agent):
    def __init__(self, **kwargs):
        # Set up houses
        self.houses = feeder_loads
        kwargs['result_path'] = feeder_results_path
        print('_feeder loads')
        print(feeder_loads)
        print('_self.houses')
        print(self.houses)

        self.feeder = None
        self.all_load_names = []
        self.all_bus_names = []

        super().__init__('Feeder', **kwargs)

    def initialize(self):
        # create feeder instance
        redirects = [master_dss_file]
        self.feeder = OpenDSS(redirects, freq_house, start_time)

        self.all_load_names = self.feeder.get_all_elements(element='Load').index
        self.all_bus_names = self.feeder.get_all_buses()

        # set up results files
        self.initialize_results('community_P')
        self.initialize_results('community_Q')
        self.initialize_results('community_V')
        self.initialize_results('summary')
        self.initialize_results('all_voltages')
        self.initialize_results('all_powers')

    def setup_pub_sub(self):
        print('_DOOM subscriptions')
        print(self.houses)
        for house in self.houses:
            topic_from_house = "house_{}_power_to_feeder".format(house)
            self.register_sub(house, topic_from_house)

            topic_to_house = "feeder_vtg_to_house_{}".format(house)
            self.register_pub(house, topic_to_house)

    def setup_actions(self):
        self.add_action(self.run_feeder, 'Run Feeder', freq_house, offset_feeder_run)
        self.add_action(self.save_results, 'Save Results', freq_save_results, offset_save_results)

    def get_house_power(self, house):
        data = self.fetch_subscription(house)
        if data is not None:
            p = data['P Total']
            q = data['Q Total']
            return p, q
        else:
            return None, None

    def send_voltage_to_houses(self):
        house_voltages = {}
        for house, load in self.houses.items():
            v = self.feeder.get_voltage(load, average=True)
            house_voltages[house] = v

            # Send voltage to house
            self.publish_to_topic(house, {"voltage": v})
        return house_voltages

    def add_community_results(self, house_powers, house_voltages):
        # save powers and voltages to results files

        p = {house: house_powers.get(house, (None, None))[0] for house in self.houses}
        self.add_to_results('community_P', p, remove_seconds=True)

        q = {house: house_powers.get(house, (None, None))[1] for house in self.houses}
        self.add_to_results('community_Q', q, remove_seconds=True)

        v = {house: house_voltages.get(house, 0) for house in self.houses}
        self.add_to_results('community_V', v, remove_seconds=True)

        # return total p and q
        try:
            return sum(p.values()), sum(q.values())
        except TypeError:
            return 0, 0

    def run_feeder(self):
        house_powers = {}
        # Get house powers and update feeder
        print('_houses')
        print(self.houses)
        for house, load in self.houses.items():
            p, q = self.get_house_power(house)
            house_powers[house] = p, q
            print('_house powers: {}, {}'.format(p, q))
            if p is not None:
                self.feeder.set_power(load, p, q)

        # Run DSS
        self.feeder.run_dss()

        # Get house voltages and send to houses
        house_voltages = self.send_voltage_to_houses()
        if house_voltages:
            v_max = max(house_voltages.values())
            v_min = min(house_voltages.values())
            self.print_log('Community Voltage Range: {} - {}'.format(v_min, v_max))

            # Add results for each house
        p_total, q_total = self.add_community_results(house_powers, house_voltages)

        # Add substation results, including community Loads, PV, and Battery
        substation_data = self.feeder.get_circuit_info()
        substation_data['Community Loads P (MW)'] = p_total / 1000
        substation_data['Community Loads Q (MVAR)'] = q_total / 1000
        self.add_to_results('summary', substation_data, remove_seconds=True)

        # Add results for all load powers and voltages
        data = self.feeder.get_all_bus_voltages(average=True)
        self.add_to_results('all_voltages', data, remove_seconds=True)

        all_powers = {load: self.feeder.get_power(load, total=True) for load in self.all_load_names}
        # print('_all powers')
        # print(all_powers)
        all_powers = {load + pq: val for load, powers in all_powers.items() for pq, val in
                      zip(('_P', '_Q'), powers)}
        self.add_to_results('all_powers', all_powers, remove_seconds=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        addr = str(sys.argv[1])
        agent = Feeder(broker_addr=addr)
    else:
        agent = Feeder(debug=True, run_helics=False)

    agent.simulate()
