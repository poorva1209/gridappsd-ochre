# -*- coding: utf-8 -*-
import helics as h
import json
import time
from pandas import json_normalize
import cffi

from constants import *


def make_helics_federate(name, broker_addr=None, fed_type='value', **kwargs):
    if broker_addr != None and kwargs.get('simulation_id',None) != None:
        fedinitstring = "--federates=1 --broker_address=tcp;//{}".format(kwargs.get('simulation_id',""))
    if broker_addr is None or broker_addr == '0.0.0.0:4545':
        fedinitstring = "--federates=1"
    else:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ipaddr = s.getsockname()[0]
        fedinitstring = "--federates=1 --broker_address=tcp://{} --interface=tcp://{}".format(
            broker_addr, ipaddr)

    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, name)
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring)
    h.helicsFederateInfoSetTimeProperty(
        fedinfo, h.helics_property_time_delta, time_step.total_seconds())
    if fed_type == "value":
        return h.helicsCreateValueFederate(name, fedinfo)
    elif fed_type == "message":
        return h.helicsCreateMessageFederate(name, fedinfo)
    else:
        return h.helicsCreateCombinationFederate(name, fedinfo)


class Agent:
    def __init__(self, name, run_helics=True, **kwargs):
        self.name = name
        self.debug = debug
        self.current_time = start_time
        self.time_step = time_step

        # Set up Actions and Results
        self.actions = []
        self.setup_actions()
        self.results = {}
        self.print_log(kwargs.get('result_path'))
        self.result_path = kwargs.get(
            'result_path', os.path.join(output_path, self.name))
        self.print_log(self.result_path)
        os.makedirs(self.result_path, exist_ok=True)

        # Initialize Agent
        t = datetime.now()
        self.print_log('Initializing Agent...')
        self.initialize()
        self.print_log('Time to initialize:', datetime.now() - t)

        # Connect to Helics
        if run_helics:
            self.print_log('Creating Federate (Helics version: {})'.format(
                h.helicsGetVersion()))
            self.fed = make_helics_federate(name, **kwargs)
        else:
            self.fed = None

        # Set up publications and subscriptions
        self.publications = {}
        self.subscriptions = {}
        self.endpoints = {}
        self.setup_pub_sub()

        if run_helics:
            # Finish federate initialization
            h.helicsFederateEnterInitializingMode(
                self.fed)  # This isn't necessary
            self.print_log(
                'Federate initialized. Waiting for other federates...')
            h.helicsFederateEnterExecutingMode(self.fed)
            self.print_log('Federate entering execution mode')

        self.print_log('Beginning Co-simulation')

    def add_action(self, func, name, freq, offset=timedelta(seconds=0)):
        action = {'func': func,
                  'name': name,
                  'freq': freq,
                  'offset': offset,
                  'times': []}
        self.actions.append(action)

    def register_pub(self, name, topic=None, var_type=h.helics_data_type_string, global_type=True, include_results=True):
        if topic is None:
            topic = name
        if self.fed is not None:
            if global_type:
                pub = h.helicsFederateRegisterGlobalTypePublication(
                    self.fed, topic, var_type, "")
            else:
                pub = h.helicsFederateRegisterPublication(
                    self.fed, topic, var_type, "")
        else:
            pub = None
        if include_results and var_type == 'String':
            self.initialize_results(name)
        else:
            include_results = False

        self.publications[name] = (pub, var_type, include_results)

    def register_sub(self, name, topic=None, var_type=h.helics_data_type_string, default=None):
        if topic is None:
            topic = name
        if self.fed is not None:
            sub = h.helicsFederateRegisterSubscription(self.fed, topic, "")
        else:
            sub = None
        self.subscriptions[name] = (sub, var_type, default)

    def register_endpoint(self, name, topic=None, var_type='string'):
        if topic is None:
            topic = name
        if self.fed is not None:
            ep = h.helicsFederateRegisterEndpoint(self.fed, topic, var_type)
        else:
            ep = None
        self.endpoints[name] = (ep, var_type)

    def publish_to_topic(self, name, value):
        if self.fed is None:
            return
        if self.debug:
            self.print_log('Publishing on {}:'.format(name), value)

        pub, var_type, has_results = self.publications[name]
        if var_type == h.helics_data_type_string:
            msg = json.dumps(value)
            h.helicsPublicationPublishString(pub, msg)
        elif var_type == h.helics_data_type_complex:
            h.helicsPublicationPublishComplex(pub, value.real, value.imag)
        else:
            h.helicsPublicationPublishDouble(pub, value)

        if has_results:
            self.add_to_results(name, value)

    def fetch_subscription(self, name):
        sub, var_type, default = self.subscriptions[name]
        if self.fed is None:
            return default
        is_updated = h.helicsInputIsUpdated(sub)
        if not is_updated:
            self.print_log('No new data from subscription:', name)
            return default

        if var_type == h.helics_data_type_string:
            data = h.helicsInputGetString(sub)
            if not isinstance(default, str):
                data = json.loads(data)
        elif var_type == h.helics_data_type_complex:
            data = h.helicsInputGetComplexObject(sub)
        else:
            data = h.helicsInputGetDouble(sub)

        if self.debug:
            self.print_log('Received from {}:'.format(name), data)
        return data

    def fetch_messages(self, name):
        ep, _ = self.endpoints[name]
        if self.fed is None or ep is None:
            return None
        pendingMessageCount = h.helicsEndpointPendingMessagesCount(ep)
        messageDicts = []
        for _ in range(pendingMessageCount):
            message = h.helicsEndpointGetMessage(ep)
            messageString = h.helicsMessageGetString(message)
            messageDict = json.loads(messageString)
            messageDicts.append(messageDict)
            if self.debug:
                self.print_log('Received message from {}:'.format(name), messageDict)
    
        return messageDicts

    def step_to(self, requested_time):
        t_requested = (requested_time - start_time).total_seconds()

        if self.fed is not None:
            t_current = (self.current_time - start_time).total_seconds()
            while t_current < t_requested:
                t_current = h.helicsFederateRequestTime(self.fed, t_requested)

        self.current_time = requested_time
        # if self.debug:
        #     self.print_log('Granted time: {}'.format(requested_time))
        return

    def setup_pub_sub(self):
        # Example: self.register_pub(load, topic_to_load, h.helics_data_type_string)
        #          self.register_sub(load, topic_from_load)
        raise NotImplementedError

    def setup_actions(self):
        # Example: self.add_action(func, name, freq, offset)
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def finalize(self):
        self.print_log('Saving Final Results...')
        self.save_results()

        for action in self.actions:
            x = action['times']
            if x:
                msg = '{} ran {} times: Avg. Time: {}, Total Time: {}, Max. Time: {}'
                self.print_log(msg.format(action['name'], len(
                    x), sum(x) / len(x), sum(x), max(x)))

        if self.fed is not None:
            self.print_log('Freeing federate')
            h.helicsFederateFinalize(self.fed)
            h.helicsFederateFree(self.fed)
            h.helicsCloseLibrary()

    def initialize_results(self, result_name, result_file=None):
        if result_file is None:
            result_file = os.path.join(
                self.result_path, '{}_{}.csv'.format(self.name, result_name))
        if os.path.exists(result_file):
            os.remove(result_file)
        self.results[result_name] = {'filename': result_file,
                                     'data': []}

    def add_to_results(self, result_name, data, add_time=True, remove_microseconds=True, remove_seconds=False):
        if add_time:
            t = self.current_time
            if remove_microseconds:
                t = t.replace(microsecond=0)
            if remove_seconds:
                t = t.replace(second=0)
            tmp = {'Time': t}
            tmp.update(data)
            data = tmp
        self.results[result_name]['data'].append(data)

    def save_results(self):
        # save data to file and remove from results
        for name, results in self.results.items():
            df = pd.DataFrame(json_normalize(results['data']))
            if not len(df):
                continue

            result_file = results['filename']
            self.print_log(
                'Saving {} results to: {}'.format(name, result_file))
            if os.path.exists(result_file):
                # append results
                df.to_csv(result_file, mode='a', header=False, index=False)
            else:
                df.to_csv(result_file, header=True, index=False)

            # remove results
            self.results[name]['data'] = []

    def simulate(self):
        for t in times:
            self.step_to(t)
            dt = t - start_time

            for action in self.actions:
                offset = action['offset']
                freq = action['freq']
                if (dt - offset) % freq == timedelta(0):
                    if self.debug:
                        self.print_log('Running action: ' + action['name'])
                    t_start = datetime.now()
                    try:
                        action['func']()
                    except Exception as e:
                        self.print_log('ERROR - Co-simulation Failed')
                        self.finalize()
                        raise e
                    t_end = datetime.now()
                    action['times'].append((t_end - t_start).total_seconds())

        self.finalize()
        for action in self.actions:
            x = action['times']
            if x:
                msg = '{} ran {} times: Avg. Time: {}, Total Time: {}, Max. Time: {}'
                self.print_log(msg.format(action['name'], len(
                    x), sum(x) / len(x), sum(x), max(x)))

    def print_log(self, *args):
        print('{} - {} at {}:'.format(datetime.now(),
                                      self.name, self.current_time), *args)
