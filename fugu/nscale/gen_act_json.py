'''network_fanin
read in the networks/network.json file that contains the network information 
read in the networks/network/fanin.txt that contains the partitioning information
generate the network_act.json file as the input for ACT simulation
'''

import json
import act_parameters as para
  
f = open('networks/network.json')
network = json.load(f)

f_fanin = open('networks/network_fanin.json')
network_fanin = json.load(f_fanin)

# cluster sizes (cluster_size_x, cluster_size_y)
cluster_size_x = 0
cluster_size_y = 0
for i in network_fanin["neuron_dict"]:
    if network_fanin["neuron_dict"][i]["core"][0] > cluster_size_x:
        cluster_size_x = network_fanin["neuron_dict"][i]["core"][0]
    if network_fanin["neuron_dict"][i]["core"][1] > cluster_size_y:
        cluster_size_y = network_fanin["neuron_dict"][i]["core"][1]
cluster_size_x += 1
cluster_size_y += 1

# each cluster has 2x2 cores
core_size_x = 2
core_size_y = 2

cores_list = []
for i in range(cluster_size_x):
    l_2 = []
    for j in range(cluster_size_y):
        l_3 = []
        for k in range(core_size_x * core_size_y):
            l_3.append({})
        l_2.append(l_3)
    cores_list.append(l_2)

for i in range(cluster_size_x):
    for j in range(cluster_size_y):
        for k in range(core_size_x * core_size_y):
            cores_list[i][j][k]["x"] = i
            cores_list[i][j][k]["y"] = j
            cores_list[i][j][k]["id"] = k
            cores_list[i][j][k]["remote"] = []
            cores_list[i][j][k]["neuron_models"] = []
            cores_list[i][j][k]["input_axon"] = []
            cores_list[i][j][k]["neurons"] = []

# the neuron models that already stored in each core
# neuron_model = (th, leak, bias, refractory period, prob, init_potential)
# {neuron_model: model_id} 
cores_list_neuron_model = []

# the remote cores that already stored in each core
# remote = (cluster_x, cluster_y, core_id)
# {remote: remote_index} 
cores_list_remote = []

for i in range(cluster_size_x):
    l_2 = []
    for j in range(cluster_size_y):
        l_3 = []
        for k in range(core_size_x * core_size_y):
            l_3.append({})
        l_2.append(l_3)
    cores_list_neuron_model.append(l_2)

for i in range(cluster_size_x):
    l_2 = []
    for j in range(cluster_size_y):
        l_3 = []
        for k in range(core_size_x * core_size_y):
            l_3.append({})
        l_2.append(l_3)
    cores_list_remote.append(l_2)  

# axon dict
# each neuron fanouts to the same axon in each core
# axon_dict{pre-synaptic neuron:{(core_id): axon_id}}
axon_dict = {}

for i in network["neuron_dict"]:
    neuron_model = (int(network["neuron_dict"][i]["th"]),
                    int(network["neuron_dict"][i]["leak"]),
                    int(network["neuron_dict"][i]["bias"]),
                    network["neuron_dict"][i]["refractory period"],
                    int(network["neuron_dict"][i]["prob"]))
    core_id = network_fanin["neuron_dict"][i]["core"]
    network["neuron_dict"][i]["neuron_id"] = len(cores_list[core_id[0]][core_id[1]][core_id[2]]["neurons"]) 
    network_fanin["neuron_dict"][i]["neuron index"] = network["neuron_dict"][i]["neuron_id"]
    if neuron_model not in cores_list_neuron_model[core_id[0]][core_id[1]][core_id[2]]:     
        cores_list_neuron_model[core_id[0]][core_id[1]][core_id[2]][neuron_model] = len(cores_list_neuron_model[core_id[0]][core_id[1]][core_id[2]]) + 1
        cores_list[core_id[0]][core_id[1]][core_id[2]]["neuron_models"].append({"th": neuron_model[0], "leak": neuron_model[1], "bias": neuron_model[2], "refractory": neuron_model[3]})
    cores_list[core_id[0]][core_id[1]][core_id[2]]["neurons"].append({"model": cores_list_neuron_model[core_id[0]][core_id[1]][core_id[2]][neuron_model], "fanout": []}) 
    axon_dict[i] = {}

for i in network["synapse_dict"]:
    post_neuron_core = tuple(network_fanin["neuron_dict"][i]["core"])
    post_neuron_id = network["neuron_dict"][i]["neuron_id"] # the neuron id in its core      
    for j in network["synapse_dict"][i]:
        if j != "-1": # neuron-to-neuron synapse
            pre_neuron_core = tuple(network_fanin["neuron_dict"][j]["core"])
            pre_neuron_id = network["neuron_dict"][j]["neuron_id"]
            if post_neuron_core not in axon_dict[j]:
                axon_dict[j][post_neuron_core] = len(cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"])
                cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"].append([])               
            if post_neuron_core not in cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]:
                cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]][post_neuron_core] = len(cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]])
                cores_list[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]["remote"].append({"x": post_neuron_core[0], "y": post_neuron_core[1], "id": post_neuron_core[2]}) 
            if {"remote": cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]][post_neuron_core], "axon": axon_dict[j][post_neuron_core]} not in cores_list[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]["neurons"][pre_neuron_id]["fanout"]:
                cores_list[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]["neurons"][pre_neuron_id]["fanout"].append({"remote": cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]][post_neuron_core], "axon": axon_dict[j][post_neuron_core]})
            for k in network["synapse_dict"][i][j]:
                if "wt" in network["synapse_dict"][i][j][k]:
                    cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"][axon_dict[j][post_neuron_core]].append({"neuron": post_neuron_id, "wt": network["synapse_dict"][i][j][k]["wt"]})
                elif "fixed-wt" in network["synapse_dict"][i][j][k]: 
                    cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"][axon_dict[j][post_neuron_core]].append({"neuron": post_neuron_id, "fixed-wt": network["synapse_dict"][i][j][k]["fixed-wt"]})
        else: # external synapse
            for k in network["synapse_dict"][i][j]:
                cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"].append([])
                cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"][len(cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"]) - 1].append({"neuron": post_neuron_id, "fixed-wt": network["synapse_dict"][i][j][k]["fixed-wt"]})
                network["synapse_dict"][i][j][k]["axon_id"] = len(cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"]) - 1

network_act = {}
network_act["ticks"] = network["ticks"]  

network_act['cores'] = []
for i in range(cluster_size_x):
    for j in range(cluster_size_y):
        for k in range(core_size_x * core_size_x):
            network_act['cores'].append(cores_list[i][j][k])

if "spikes_dict" in network:
    network_act["spikes"] = []
    for i in network["spikes_dict"]:
        network_act["spikes"].append({"tick": int(i), "cores": []})
        for j in network["spikes_dict"][i]["syn"]:
            neuron_core_id = network_fanin["neuron_dict"][str(j[0])]["core"]
            network_act["spikes"][len(network_act["spikes"]) - 1]["cores"].append({
                "x": neuron_core_id[0],
                "y": neuron_core_id[1],
                "id": neuron_core_id[2],
                "input_axon": [network["synapse_dict"][str(j[0])][str(j[1])][str(j[2])]["axon_id"]]
            })


max_num_neuron = 0
max_neuron_fanout = 0
max_size_fanout_table = 0
max_num_axon = 0
max_num_neuron_model = 0
max_num_remote = 0
max_size_axon_table = 0
max_num_fanin = 0

for i in network_act["cores"]:
    neuron_fanout = 0
    size_fanout_table = 0
    size_axon_table = 0

    num_neuron = len(i["neurons"])
    num_axon = len(i["input_axon"])
    num_remote = len(i["remote"])
    num_neuron_model = len(i["neuron_models"])
    
    for j in i["input_axon"]:
        size_axon_table += len(j)
    size_axon_table += num_axon
    
    for j in i["neurons"]:
        size_fanout_table += len(j['fanout'])
        neuron_fanout = max(neuron_fanout, len(j['fanout']))
    size_fanout_table += num_neuron
    
    max_num_neuron = max(max_num_neuron, num_neuron)
    max_neuron_fanout = max(max_neuron_fanout, neuron_fanout)
    max_size_fanout_table = max(max_size_fanout_table, size_fanout_table)
    max_num_axon = max(max_num_axon, num_axon)
    max_num_neuron_model = max(max_num_neuron_model, num_neuron_model)
    max_num_remote = max(max_num_remote, num_remote)
    max_size_axon_table = max(max_size_axon_table, size_axon_table)

for i in network_act["cores"]:  
    num_fanin = 0
    for fanin_coreinfo in network_act['cores']:
        if {'x': i["x"], 'y':i["y"], 'id':i["id"]} in fanin_coreinfo['remote']:
            num_fanin += 1
    max_num_fanin = max(max_num_fanin, num_fanin)

#assert cluster_size_x <= para.GRID_X, f"GRID_X should not be larger than {para.GRID_X}, got: {cluster_size_x}"
#assert cluster_size_y <= para.GRID_Y, f"GRID_Y should not be larger than {para.GRID_Y}, got: {cluster_size_y}"
assert max_num_neuron <= para.NEURONS_PER_CORE, f"NEURONS_PER_CORE should not be larger than {para.NEURONS_PER_CORE}, got: {max_num_neuron}"
assert max_size_fanout_table <= para.NEURON_FANOUT_TABLE, f"NEURON_FANOUT_TABLE should not be larger than {para.NEURON_FANOUT_TABLE}, got: {max_size_fanout_table}"
assert max_num_axon <= para.INPUT_AXONS_PER_CORE, f"INPUT_AXONS_PER_CORE should not be larger than {para.INPUT_AXONS_PER_CORE}, got: {max_num_axon}"
assert max_size_axon_table <= para.INPUT_AXON_TABLE_SZ, f"INPUT_AXON_TABLE_SZ should not be larger than {para.INPUT_AXON_TABLE_SZ}, got: {max_size_axon_table}"
assert max_neuron_fanout <= para.MAX_NEURON_FANOUT, f"MAX_NEURON_FANOUT should not be larger than {para.MAX_NEURON_FANOUT} expected, got: {max_neuron_fanout}"
assert max_num_remote <= para.REMOTE_CORES, f"REMOTE_CORES should not be larger than {para.REMOTE_CORES}, got: {max_num_remote}"
assert (max_num_neuron_model+1) <= para.NEURON_MODELS, f"NEURON_MODELS should not be larger than {para.NEURON_MODELS}, got: {(max_num_neuron_model+1)}"
assert max_num_fanin <= para.LEN_FANIN_HT, f"LEN_FANIN_HT should not be larger than {para.LEN_FANIN_HT}, got: {max_num_fanin}"

f_act = 'act_json/act_network.json'
with open(f_act, 'w') as f_act:
    json.dump(network_act, f_act)  

# dump neuron index into network_fanin_updated.json
f_fanin_updated = 'networks/network_fanin_updated.json'
with open(f_fanin_updated, 'w') as f_fanin_updated:
    json.dump(network_fanin, f_fanin_updated)  

print("successfully generate act_json/act_network.json")

# write done the expected grid size
f_path = 'expected_grid_size.txt'
with open(f_path, 'w') as f:
    f.write(str(cluster_size_x) + ' x ' + str(cluster_size_y) + ' gridded clusters.\n')