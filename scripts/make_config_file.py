import shutil
import sys
import math
import json
import socket
import subprocess
from copy import copy
from datetime import datetime
from constants import *

import cProfile
import os


# Get the host type 
host = sys.argv[1]
if BTO_sim:
    houses_per_broker = 10
else:
    houses_per_broker = 30

# Set paths 
config_file = os.path.join(base_path, "bin", "config.json")
federates_directory = os.path.join(base_path, 'agents')

# config skeleton
config = {"broker": host == "localhost",
          "federates": [],
          "name": scenario_name,
          }
federate = {"directory": federates_directory,
            "host": host,
            }

# To facilitate launching aggregators in a seperate node
aggre_config = {"broker": host == "localhost",
                "federates": [],
                "name": scenario_name,
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
        print(splt)
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

    if include_aggregator:
        no_of_config_file += 1
        fed_len += len(aggre_config["federates"])

    # creating config file
    for i in range(1, no_of_config_file + 1):
        with open("{}/config_{}.json".format(output_path, i), "w+") as f1:
            data = {"broker": "false",
                    "name": config["name"]
                    }
            if i == no_of_config_file and include_aggregator:
                data["federates"] = aggre_config['federates']
            else:
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
    ip_addr = '0.0.0.0:4545'
else:
    nodes_count, node_list = get_nodes_info()
    # TODO: test and remove
    # _,ip_addr = ssh_nodes(["ssh", node_list[0], "-f",'cd /projects/qstsai/SETO/Co-sim;python bin/ip.py'])
    # ip_addr = str(ip_addr)[1:]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_addr = str(s.getsockname()[0])
    ip_addr += ":4545"

print("IP addr", ip_addr)

# Creating the feeder fed
if include_feeder:
    feeder = copy(federate)
    feeder['name'] = "feeder"
    feeder['exec'] = "python Feeder.py {}".format(ip_addr)
    config['federates'].append(feeder)


# Add houses, hems, and brokers
for i, load in enumerate(house_ids):
    # add house and hems
    house = copy(federate)
    house['exec'] = "python House.py {} {}".format(load, ip_addr)
    house['name'] = "house_{}".format(load)
    config['federates'].append(house)

    if include_hems:
        hems = copy(federate)
        hems['exec'] = "python Hems.py {} {}".format(load, ip_addr)
        hems['name'] = "hems_{}".format(load)
        config['federates'].append(hems)


# Creating aggregator federates for each PCC
if include_aggregator:
    for pcc in aggregator_pccs:
        agg = copy(federate)
        agg['name'] = "aggregator_{}".format(pcc)
        agg['exec'] = "python Aggregator.py {} {}".format(pcc, ip_addr)
        if host == "eagle":
            aggre_config['federates'].append(agg)
        else:
            config['federates'].append(agg)

# Creating utility federate
if include_utility:
    utility = copy(federate)
    utility['name'] = "utility"
    utility['exec'] = "python Utility.py {}".format(ip_addr)
    if host == "eagle":
        aggre_config['federates'].append(utility)
    else:
        config['federates'].append(utility)

# Create the output directory for the scenario
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
os.makedirs(aggregator_results_path, exist_ok=True)
os.makedirs(hems_results_path, exist_ok=True)
os.makedirs(house_results_path, exist_ok=True)

# save config to the main config file
if host == "localhost":
    with open(config_file, 'w+') as f:
        f.write(json.dumps(config, indent=4, sort_keys=True, separators=(",", ":")))
        cmd = "helics run --path bin/config.json --broker-loglevel=7"
        print(cmd)
elif host == "eagle":
    # Record the start time of the simulation
    start_time = datetime.now()

    (fed_len, config_files, out_files, err_files) = construct_configs()
    print("Number of federates", fed_len)
    print("config files", config_files)

    # Start the helics runner
    cmd = "helics_broker -f {} --interface=tcp://0.0.0.0:4545 --loglevel=5".format(fed_len)
    print(cmd)
    p1 = subprocess.Popen(cmd.split(" "))

    # ssh into each node and run the config files
    for i in range(len(config_files)):
        if 'BTO' in scenario_name:
            cmd = "cd {};source bin/scenario_BTO_{}_{}.sh;python {}/bin/run_agents_from_helicsrunner.py {}/{}" \
                .format(base_path, scenario_name.split('_')[2], season, base_path, output_path, config_files[i])
        else:
            cmd = "cd {};source bin/scenario_{}_{}.sh;python {}/bin/run_agents_from_helicsrunner.py {}/{}" \
                .format(base_path, scenario_name.split('_')[1], season, base_path, output_path, config_files[i])

        # For debugging purpose records the ssh commands
        file_red = open(os.path.join(output_path, "{}_outfile.txt".format(i)), 'a')
        ssh_cmd = "ssh {} -f {}".format(node_list[i + 1], cmd)

        with open(config_file, 'w+') as f:
            file_red.write(json.dumps(ssh_cmd))
            file_red.close()


        ssh = subprocess.Popen(ssh_cmd.split(' '),
                               shell=False,
                               stdout=out_files[i],
                               stderr=err_files[i])

    p1.wait()

    print("Time taken to finish the simulation: ", datetime.now() - start_time)
