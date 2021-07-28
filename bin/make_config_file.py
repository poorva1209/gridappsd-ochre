import shutil
import sys
import os
import math
import json
import socket
import subprocess
import cProfile
from copy import copy
from datetime import datetime
from constants import *

# Get the host type 
host = sys.argv[1]
base_path = sys.argv[2]
config_file_name = sys.argv[3]
port = sys.argv[4]
simulation_id = sys.argv[5]

houses_per_broker = 60

# Set paths 
config_file = os.path.join(base_path, config_file_name)
federates_directory = os.path.join(base_path, 'agents')
output_path = os.path.join(base_path, "outputs")

# config skeleton
# Never start HELICS broker, it will be started by the GridAPPS-D platform
config = {
    "broker": False,
    "federates": [],
    "name": scenario_name
}

federate = {
    "directory": federates_directory,
    "host": host
}

gld_path = os.path.join(base_path, "inputs", "gridlabd", 'IEEE-13')
gld_model = 'IEEE-13_Houses.glm'
gld_helics_config = {
    "name": str(simulation_id),
    "loglevel": 3,
    "broker": f"127.0.0.1:{port}",
    "coreType": "zmq",
	"coreName": f"{simulation_id}_core",
    "period": 1.0,
    "uninterruptible": False,
    "publications": [],
    "subscriptions": [],
    "endpoints": [
        {
            "name": "helics_input",
            "global": False,
            "type": "string",
            "info": '{"message_type": "JSON"}'
        },
        {
            "name": "helics_output",
            "global": False,
            "type": "string",
            "destination": f"HELICS_GOSS_{simulation_id}/helics_output",
            "info": '{"message_type": "JSON","publication_info": {"node_684": ["voltage_C","voltage_A"],"ld_tl_house_23_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"node_680": ["voltage_B","voltage_C","voltage_A"],"ld_tl_house_38_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"ld_tl_house_7_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_6_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"swt_j1": ["current_in_C","status","current_in_B","current_in_A"],"ld_tl_house_10_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"swt_j2": ["status","current_in_B","current_in_C","current_in_A"],"rootbus": ["voltage_B","voltage_A","voltage_C"],"ld_tl_house_40_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_24_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"line_684-611": ["power_in_C"],"ld_tl_house_5_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_8_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"node_675": ["voltage_B","voltage_C","voltage_A"],"tpx_object_110": ["power_in_B","power_in_A"],"ld_tl_house_39_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"tpx_object_111": ["power_in_B","power_in_A"],"tpx_object_112": ["power_in_B","power_in_A"],"tpx_object_113": ["power_in_A","power_in_B"],"ld_tl_house_22_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"node_671": ["voltage_C","voltage_A","voltage_B"],"tpx_object_114": ["power_in_B","power_in_A"],"ld_tl_house_25_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tpx_object_115": ["power_in_A","power_in_B"],"tpx_object_116": ["power_in_A","power_in_B"],"tpx_object_117": ["power_in_A","power_in_B"],"tpx_object_118": ["power_in_A","power_in_B"],"tpx_object_119": ["power_in_A","power_in_B"],"ld_tl_house_36_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"swt_trunk": ["status","current_in_A","current_in_B","current_in_C"],"cap_cap1": ["voltage_A","switchB","switchC","shunt_C","voltage_C","switchA","voltage_B","shunt_A","shunt_B"],"cap_cap2": ["voltage_B","voltage_C","switchB","shunt_C","shunt_B","voltage_A","shunt_A","switchC","switchA"],"ld_tl_house_11_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"line_630-632": ["power_in_A","power_in_C","power_in_B"],"tl_house_4": ["voltage_1","voltage_2"],"sourcebus": ["voltage_A","voltage_B","voltage_C"],"tl_house_5": ["voltage_1","voltage_2"],"tl_house_2": ["voltage_1","voltage_2"],"tpx_object_120": ["power_in_B","power_in_A"],"tpx_object_121": ["power_in_B","power_in_A"],"tl_house_3": ["voltage_2","voltage_1"],"tpx_object_122": ["power_in_A","power_in_B"],"ld_tl_house_18_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tl_house_1": ["voltage_1","voltage_2"],"tpx_object_123": ["power_in_B","power_in_A"],"tpx_object_124": ["power_in_B","power_in_A"],"tpx_object_125": ["power_in_B","power_in_A"], "tpx_object_126": ["power_in_B","power_in_A"],"tpx_object_127": ["power_in_B","power_in_A"],"tpx_object_128": ["power_in_A","power_in_B"],"tpx_object_129": ["power_in_A","power_in_B"],"ld_tl_house_4_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_12_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"ld_tl_house_21_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"line_632-6321": ["power_in_A","power_in_B","power_in_C"],"ld_tl_house_27_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"line_632-633": ["power_in_B","power_in_A","power_in_C"],"trip_node15": ["voltage_1","voltage_2"],"tl_house_8": ["voltage_1","voltage_2"],"tl_house_9": ["voltage_2","voltage_1"],"tl_house_6": ["voltage_1","voltage_2"],"tl_house_7": ["voltage_2","voltage_1"],"node_650": ["voltage_B","voltage_A","voltage_C"],"tpx_object_130": ["power_in_B","power_in_A"],"tpx_object_131": ["power_in_B","power_in_A"],"node_652": ["voltage_A"],"tpx_object_132": ["power_in_A","power_in_B"],"tpx_object_133": ["power_in_A","power_in_B"],"tpx_object_134": ["power_in_A","power_in_B"],"tpx_object_135": ["power_in_B","power_in_A"],"tpx_object_136": ["power_in_A","power_in_B"],"tpx_object_137": ["power_in_A","power_in_B"],"tpx_object_138": ["power_in_B","power_in_A"],"tpx_object_139": ["power_in_A","power_in_B"],"ld_tl_house_31_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_34_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"ld_tl_house_37_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"line_632-645": ["power_in_C","power_in_B"],"line_645-646": ["power_in_B","power_in_C"],"node_6321": ["voltage_B", "voltage_C","voltage_A"],"tpx_object_140": ["power_in_B","power_in_A"],"tpx_object_141": ["power_in_B","power_in_A"],"ld_tl_house_29_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_15_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tpx_object_142": ["power_in_B","power_in_A"],"line_684-652": ["power_in_A"],"tpx_object_143": ["power_in_A","power_in_B"],"tpx_object_144": ["power_in_A","power_in_B"],"tpx_object_145": ["power_in_B","power_in_A"],"tpx_object_146": ["power_in_B", "power_in_A"],"tpx_object_147": ["power_in_B","power_in_A"],"tpx_object_148": ["power_in_B","power_in_A"],"ld_tl_house_1_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tpx_object_149": ["power_in_A","power_in_B"],"node_645": ["voltage_C","voltage_B"],"node_646": ["voltage_B","voltage_C"],"ld_tl_house_16_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"tl_house_40": ["voltage_1","voltage_2"],"ld_tl_house_32_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"ld_tl_house_30_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"trip_node10": ["voltage_1","voltage_2"],"ld_tl_house_33_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"node_630": ["voltage_C","voltage_B","voltage_A"],"trip_node13": ["voltage_1","voltage_2"],"line_671-684": ["power_in_A","power_in_C"], "trip_node14": ["voltage_2","voltage_1"], "ld_tl_house_2_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"trip_node11": ["voltage_1","voltage_2"],"trip_node12": ["voltage_1","voltage_2"],"line_671-680": ["power_in_A","power_in_C","power_in_B"],"ld_tl_house_28_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"node_632": ["voltage_C","voltage_B","voltage_A"],"node_633": ["voltage_C","voltage_B","voltage_A"],"tl_house_37": ["voltage_1","voltage_2"],"tl_house_38": ["voltage_1","voltage_2"],"tl_house_35": ["voltage_1","voltage_2"],"ld_tl_house_17_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tl_house_36": ["voltage_2","voltage_1"],"tl_house_39": ["voltage_2","voltage_1"],"ld_tl_house_14_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tl_house_30": ["voltage_1","voltage_2"],"tl_house_33": ["voltage_1","voltage_2"],"tl_house_34": ["voltage_1","voltage_2"],"tl_house_31": ["voltage_2","voltage_1"],"tl_house_32": ["voltage_2","voltage_1"],"ld_tl_house_26_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"swt_671-692": ["current_in_B","status","current_in_C","current_in_A"],"xf_source_650": ["power_in_C","power_in_A","power_in_B"],"ld_tl_house_9_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"line_692-675": ["power_in_B","power_in_C","power_in_A"],"ld_tl_house_20_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tl_house_26": ["voltage_1","voltage_2"],"tl_house_27": ["voltage_1","voltage_2"],"tl_house_24": ["voltage_1","voltage_2"],"tl_house_25": ["voltage_1","voltage_2"],"line_6321-671": ["power_in_A","power_in_B","power_in_C"],"ld_tl_house_3_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"ld_tl_house_35_240v": ["indiv_measured_power_2","indiv_measured_power_1"],"tl_house_28": ["voltage_1","voltage_2"],"tl_house_29": ["voltage_1","voltage_2"],"ld_tl_house_19_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"ld_tl_house_13_240v": ["indiv_measured_power_1","indiv_measured_power_2"],"tl_house_22": ["voltage_2","voltage_1"],"tl_house_23": ["voltage_1","voltage_2"],"tl_house_20": ["voltage_1","voltage_2"],"tl_house_21": ["voltage_2","voltage_1"],"trip_node1": ["voltage_2","voltage_1"],"trip_node2": ["voltage_2","voltage_1"],"trip_node3": ["voltage_1","voltage_2"],"node_692": ["voltage_B","voltage_A","voltage_C"],"trip_node8": ["voltage_1","voltage_2"],"trip_node9": ["voltage_2","voltage_1"],"reg_vreg1": ["tap_B","tap_C","tap_A"],"trip_node4": ["voltage_2","voltage_1"],"trip_node5": ["voltage_2","voltage_1"],"node_611": ["voltage_C"],"trip_node6": ["voltage_1","voltage_2"],"trip_node7": ["voltage_1","voltage_2"],"tl_house_15": ["voltage_1","voltage_2"],"tl_house_16": ["voltage_1","voltage_2"],"tl_house_13": ["voltage_1","voltage_2"],"tl_house_14": ["voltage_1","voltage_2"],"tl_house_19": ["voltage_2","voltage_1"],"tl_house_17": ["voltage_1","voltage_2"],"tl_house_18": ["voltage_2","voltage_1"],"tl_house_11": ["voltage_2","voltage_1"],"tl_house_12": ["voltage_1","voltage_2"],"tl_house_10": ["voltage_1","voltage_2"],"globals": ["clock"]}}'
        }
    ]
}


# List the nodes allocated on job for ssh
def get_nodes_info():
    nodes_count = int(os.environ['SLURM_NNODES'])
    nodes = os.environ['SLURM_JOB_NODELIST']
    node_list = []
    node_st = ""

    def iterate_range(node_st, start=None):
        if start is None:
            if '-' in node_st:
                prefix = node_st.split('[')[0]
                st = node_st.split('[')[1].replace(']', '').split('-') if ']' in node_st else node_st.split('[')[
                    1].split('-')
            [node_list.append(prefix + str(no)) for no in range(int(st[0]), int(st[1]) + 1)]
        else:
            node_list.append(node_st + str(start))

    for splt in nodes.split(','):
        if '-' in splt:
            st = splt if '[' in splt else node_st + '[' + splt
            node_st = splt.split('[')[0] if '[' in splt else node_st
            iterate_range(st)
        elif '[' in splt:
            temp = splt.split('[')
            node_st = temp[0]
            iterate_range(node_st, temp[1])
        elif ']' in splt:
            iterate_range(node_st, splt.replace(']', ''))
        else:
            if 'r' not in splt:
                iterate_range(node_st, splt)
            else:
                node_list.append(splt)

    print(node_list)
    return nodes_count, node_list


# Execute commands by ssh-ing into the node
def ssh_nodes(cmd_list):
    ssh = subprocess.Popen(cmd_list,
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           preexec_fn=os.setsid)
    print("process running", cmd_list)
    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        print("SSH error", error)
    else:
        result = result[0].rstrip()
        print(result)
    return ssh, result


# Create multiple config files to run on multiple nodes
def construct_configs():
    config_files = []  # Config files for HELICS
    config_outfiles = []  # Capture the stdout on the node
    config_errfiles = []  # Capture the stderr on the node

    # Determine the number of config files needed
    federates = config["federates"]
    fed_len = len(federates)
    no_of_config_file = math.floor(fed_len / houses_per_broker)

    if len(federates) % houses_per_broker > 0:
        no_of_config_file += 1

    # creating config file
    for i in range(1, no_of_config_file + 1):
        with open("{}/config_{}.json".format(output_path, i), "w+") as f1:
            data = {
                "broker": "false",
                "name": config["name"]
            }
            federates_new_config = federates[(i - 1) * houses_per_broker:i * houses_per_broker]
            if len(federates_new_config) > 0:
                data["federates"] = federates_new_config

            if data != None:
                f1.write(json.dumps(data))
                outfile = open(os.path.join(output_path, "config_{}_outfile.txt".format(i)), 'a')
                errfile = open(os.path.join(output_path, "config_{}_errfile.txt".format(i)), 'a')
                config_files.append("config_{}.json".format(i))
                config_errfiles.append(errfile)
                config_outfiles.append(outfile)

    return (fed_len, config_files, config_outfiles, config_errfiles)


# Getting the broker's network address
if host != "eagle":
    ip_addr = '0.0.0.0:'+port
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_addr = str(s.getsockname()[0])
    ip_addr += ":"+port
print(ip_addr)
# Creating the Feeder fed
if include_feeder:
    feeder = copy(federate)
    feeder['name'] = "Feeder"
    # feeder['exec'] = "python Feeder.py {}".format(ip_addr)
    feeder['exec'] = "/gridappsd/bin/gridlabd.sh {}".format(gld_model)
    feeder['directory'] = gld_path
    config['federates'].append(feeder)

# Add houses, hems, and brokers
if include_house:
    for i, load in enumerate(house_ids):
        # add house and hems
        house = copy(federate)
        house['exec'] = "python House.py {} {} {} {}".format(load, ip_addr, simulation_id, base_path)
        house['name'] = "House_{}".format(load)
        config['federates'].append(house)
        gld_helics_config['publications'].append({
            'global': False,
            'key': 'House_{}/voltage'.format(load),
            'type': 'complex',
            'info': '{{"object" : "{}", "property" : "voltage_12"}}'.format(feeder_loads[load])
        })
        gld_helics_config['subscriptions'].append({
            'key': 'House_{}/load_to_feeder'.format(load),
            'type': 'complex',
            'required': False,
            'info': '{{"object" : "ld_{}_240v", "property" : "constant_power_12"}}'.format(feeder_loads[load])
        })

        if include_hems:
            hems = copy(federate)
            hems['exec'] = "python Hems.py {} {}".format(load, ip_addr)
            hems['name'] = "hems_{}".format(load)
            config['federates'].append(hems)
        
    with open(os.path.join(gld_path, 'gld_helics_config.json'), 'w+') as f:
        f.write(json.dumps(gld_helics_config, indent=4, separators=(",", ":")))

# Create the output directory for the scenario
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
house_results_path = os.path.join(output_path, 'Dwelling Model')
hems_results_path = os.path.join(output_path, 'Foresee')
feeder_results_path = os.path.join(output_path, 'Feeder')
os.makedirs(hems_results_path, exist_ok=True)
os.makedirs(house_results_path, exist_ok=True)
os.makedirs(feeder_results_path, exist_ok=True)

# Record the start time of the simulation
start_time = datetime.now()

# save config to the main config file
if host == "localhost":
    print(start_time)
    with open(config_file, 'w+') as f:
        f.write(json.dumps(config, indent=4, sort_keys=True, separators=(",", ":")))
        cmd = "helics run --path bin/config.json --broker-loglevel=2"
        print(cmd)

elif host == "eagle":
    nodes_count, node_list = get_nodes_info()
    (fed_len, config_files, out_files, err_files) = construct_configs()
    print("Number of federates", fed_len)
    print("config files", config_files)

    # Start the helics runner
    cmd = "helics_broker -f {} --interface=tcp://0.0.0.0:4545 --loglevel=7".format(fed_len)
    # helics_broker -f 1 --loglevel=2
    print(cmd)
    out_b = open(os.path.join(output_path, "broker_outfile.txt".format(i)), 'a')
    err_b = open(os.path.join(output_path, "broker_errfile.txt".format(i)), 'a')
    p1 = subprocess.Popen(cmd.split(" "),
         shell=False,
         stdout=out_b,
         stderr=err_b)

    # ssh into each node and run the config files
    for i in range(len(config_files)):
        cmd = "cd {}; source bin/scenario_{}.sh; python {}/bin/run_agents_from_helicsrunner.py {}/{}" \
            .format(base_path, scenario_name, base_path, output_path, config_files[i])

        # For debugging purpose records the ssh commands
        file_red = open(os.path.join(output_path, "{}_outfile.txt".format(i)), 'a')
        ssh_cmd = "ssh {} -f {}".format(node_list[i], cmd)

        with open(config_file, 'w+') as f:
            file_red.write(json.dumps(ssh_cmd))
            file_red.close()


        ssh = subprocess.Popen(ssh_cmd.split(' '),
                               shell=False,
                               stdout=out_files[i],
                               stderr=err_files[i])

    p1.wait()

    print("Time taken to finish the simulation: ", datetime.now() - start_time)
