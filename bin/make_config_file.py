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

houses_per_broker = 60

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
    ip_addr = '0.0.0.0:4545'
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_addr = str(s.getsockname()[0])
    ip_addr += ":4545"
print(ip_addr)
# Creating the Feeder fed
if include_feeder:
    feeder = copy(federate)
    feeder['name'] = "Feeder"
    # feeder['exec'] = "python Feeder.py {}".format(ip_addr)
    feeder['exec'] = "gridlabd IEEE-13_OCHRE.glm"
    feeder['directory'] = os.path.join(base_path, 'inputs', 'gridlabd')
    config['federates'].append(feeder)

# Add houses, hems, and brokers
if include_house:
    for i, load in enumerate(house_ids):
        # add house and hems
        house = copy(federate)
        house['exec'] = "python House.py {} {}".format(load, ip_addr)
        house['name'] = "House_{}".format(load)
        config['federates'].append(house)

        if include_hems:
            hems = copy(federate)
            hems['exec'] = "python Hems.py {} {}".format(load, ip_addr)
            hems['name'] = "hems_{}".format(load)
            config['federates'].append(hems)

# Create the output directory for the scenario
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
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
