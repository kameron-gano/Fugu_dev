import json
import numpy as np
from matplotlib.path import Path
import random

NSCALE_PATH = "/Users/kamerongano/Documents/GitHub/nscale/configs/distinct_demo/"

def scaffold_to_network_json(graph, timestamps, path):
    # json file
    network_json = {}
    network_json['ticks'] = timestamps
    network_json['neuron_dict'] = {}
    network_json['synapse_dict'] = {}
    
    for i, neuron in enumerate(sorted(graph.nodes)):
        neuron_info = graph.nodes[neuron]
        network_json['neuron_dict'][str(neuron_info['neuron_number'])] = {"fan-in": [],
                                                                     "fan-out": [],
                                                                     "th": neuron_info['threshold'],
                                                                     "leak": 0.0,
                                                                     "bias": 0.0,
                                                                     "refractory period": 0,
                                                                     "prob": neuron_info['p']
                                                                    }
        if 'leakage_constant' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['leak'] = neuron_info['leakage_constant']
        if 'bias' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['bias'] = neuron_info['bias'] 
        if 'refractory_period' in neuron_info:
            network_json['neuron_dict'][str(neuron_info['neuron_number'])]['refractory_period'] = neuron_info['refractory_period']    
    
    for i, synapse in enumerate(graph.edges):
        pre_neuron = graph.nodes[synapse[0]]['neuron_number']
        post_neuron = graph.nodes[synapse[1]]['neuron_number']
        network_json['neuron_dict'][str(post_neuron)]['fan-in'].append(pre_neuron)
        network_json['neuron_dict'][str(pre_neuron)]['fan-out'].append(post_neuron)
        if str(post_neuron) not in network_json['synapse_dict']:
            network_json['synapse_dict'][str(post_neuron)] = {}
        if str(pre_neuron) not in network_json['synapse_dict'][str(post_neuron)]:
            network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)] = {}
        network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)][str(len(network_json['synapse_dict'][str(post_neuron)][str(pre_neuron)]))] = {'fixed-wt': graph.edges[synapse]['weight'],
                                                                                                                                                      'delay': graph.edges[synapse]['delay']}
    with open(path, 'w') as f:
        json.dump(network_json, f) 

def generate_random_balanced_mapping(neuron_ids, num_cores=4, seed=42):
    rng = random.Random(seed)
    shuffled = list(neuron_ids)
    rng.shuffle(shuffled)

    mapping = {}
    base = len(shuffled) // num_cores
    rem = len(shuffled) % num_cores
    start = 0

    for core_id in range(num_cores):
        size = base + (1 if core_id < rem else 0)
        for neuron_id in shuffled[start:start + size]:
            mapping[neuron_id] = {
                "partition_id": core_id,
                "core": [0, 0, core_id]
            }
        start += size

    return mapping


def generate_random_mapping(neuron_ids, num_cores=4, seed=42, to_file=True, file_path='core_mappings.json'):
    rng = random.Random(seed)
    shuffled = list(neuron_ids)
    rng.shuffle(shuffled)

    mapping = {}
    for i, neuron_id in enumerate(shuffled):
        core_id = i % num_cores
        mapping[neuron_id] = {
            "partition_id": core_id,
            "core": [0, 0, core_id]
        }

    if to_file:
        core_mappings = {
            "metadata": {
                "generated": "random_round_robin",
                "seed": seed,
                "num_cores": num_cores
            },
            "neuron_dict": mapping
        }
        with open(file_path, 'w') as f:
            json.dump(core_mappings, f)

    return mapping


def load_network(path='network.json'):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path, payload, save_to_nscale=True):
    with open(path, 'w') as f:
        print(f"Writing {path}... to local directory.")
        json.dump(payload, f)
    if save_to_nscale:
        with open(NSCALE_PATH + path, 'w') as f:
            print(f"Writing {NSCALE_PATH + path}... to NeuroScale config.")
            json.dump(payload, f)


def build_core_mappings(neuron_ids, num_cores=4, seed=42):
    mapping = generate_random_balanced_mapping(neuron_ids, num_cores=num_cores, seed=seed)
    generated = "random_balanced"


    return {
        "metadata": {
            "generated": generated,
            "seed": seed,
            "num_cores": num_cores
        },
        "neuron_dict": mapping
    }


def build_network_fanin(network, core_mappings):
    mapping_neurons = core_mappings.get("neuron_dict", core_mappings)
    network_fanin = {"neuron_dict": {}}

    for neuron_id in sorted(network["neuron_dict"], key=lambda x: int(x)):
        if neuron_id not in mapping_neurons:
            raise KeyError(f"Missing mapping for neuron {neuron_id} in core_mappings")

        mapping_entry = mapping_neurons[neuron_id]
        if "core" not in mapping_entry or "partition_id" not in mapping_entry:
            raise KeyError(f"Mapping for neuron {neuron_id} must include 'core' and 'partition_id'")

        network_fanin["neuron_dict"][neuron_id] = {
            "fan-in": network["neuron_dict"][neuron_id].get("fan-in", []),
            "partition_id": mapping_entry["partition_id"],
            "core": mapping_entry["core"]
        }

    return network_fanin


def get_cluster_sizes(network_fanin):
    cluster_size_x = 0
    cluster_size_y = 0

    for neuron_id in network_fanin["neuron_dict"]:
        core = network_fanin["neuron_dict"][neuron_id]["core"]
        cluster_size_x = max(cluster_size_x, core[0])
        cluster_size_y = max(cluster_size_y, core[1])

    return cluster_size_x + 1, cluster_size_y + 1


def initialize_core_data(cluster_size_x, cluster_size_y, core_size_x=2, core_size_y=2):
    cores_list = []
    cores_list_neuron_model = []
    cores_list_remote = []

    for i in range(cluster_size_x):
        core_row = []
        model_row = []
        remote_row = []

        for j in range(cluster_size_y):
            core_group = []
            model_group = []
            remote_group = []

            for k in range(core_size_x * core_size_y):
                core_group.append({
                    "x": i,
                    "y": j,
                    "id": k,
                    "remote": [],
                    "neuron_models": [],
                    "input_axon": [],
                    "neurons": []
                })
                model_group.append({})
                remote_group.append({})

            core_row.append(core_group)
            model_row.append(model_group)
            remote_row.append(remote_group)

        cores_list.append(core_row)
        cores_list_neuron_model.append(model_row)
        cores_list_remote.append(remote_row)

    return cores_list, cores_list_neuron_model, cores_list_remote


def register_neurons(network, network_fanin, cores_list, cores_list_neuron_model):
    axon_dict = {}

    for neuron_id in sorted(network["neuron_dict"], key=lambda x: int(x)):
        neuron = network["neuron_dict"][neuron_id]
        refractory = neuron.get("refractory period", neuron.get("refractory_period", 0))
        neuron_model = (
            int(neuron["th"]),
            int(neuron["leak"]),
            int(neuron["bias"]),
            refractory,
            int(neuron["prob"])
        )

        core_id = network_fanin["neuron_dict"][neuron_id]["core"]
        core = cores_list[core_id[0]][core_id[1]][core_id[2]]
        model_registry = cores_list_neuron_model[core_id[0]][core_id[1]][core_id[2]]

        neuron_index = len(core["neurons"])
        network["neuron_dict"][neuron_id]["neuron_id"] = neuron_index
        network_fanin["neuron_dict"][neuron_id]["neuron index"] = neuron_index

        if neuron_model not in model_registry:
            model_registry[neuron_model] = len(model_registry) + 1
            core["neuron_models"].append({
                "th": neuron_model[0],
                "leak": neuron_model[1],
                "bias": neuron_model[2],
                "refractory": neuron_model[3]
            })

        core["neurons"].append({
            "model": model_registry[neuron_model],
            "fanout": []
        })
        axon_dict[neuron_id] = {}

    return axon_dict


def register_synapses(network, network_fanin, cores_list, cores_list_remote, axon_dict):
    for post_neuron in sorted(network["synapse_dict"], key=lambda x: int(x)):
        post_neuron_core = tuple(network_fanin["neuron_dict"][post_neuron]["core"])
        post_neuron_id = network["neuron_dict"][post_neuron]["neuron_id"]

        for pre_neuron, synapses in network["synapse_dict"][post_neuron].items():
            if pre_neuron != "-1":
                pre_neuron_core = tuple(network_fanin["neuron_dict"][pre_neuron]["core"])
                pre_neuron_id = network["neuron_dict"][pre_neuron]["neuron_id"]

                post_core_axons = cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"]
                pre_core_remotes = cores_list_remote[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]
                pre_core_neuron = cores_list[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]["neurons"][pre_neuron_id]

                if post_neuron_core not in axon_dict[pre_neuron]:
                    axon_dict[pre_neuron][post_neuron_core] = len(post_core_axons)
                    post_core_axons.append([])

                if post_neuron_core not in pre_core_remotes:
                    pre_core_remotes[post_neuron_core] = len(pre_core_remotes)
                    cores_list[pre_neuron_core[0]][pre_neuron_core[1]][pre_neuron_core[2]]["remote"].append({
                        "x": post_neuron_core[0],
                        "y": post_neuron_core[1],
                        "id": post_neuron_core[2]
                    })

                fanout_entry = {
                    "remote": pre_core_remotes[post_neuron_core],
                    "axon": axon_dict[pre_neuron][post_neuron_core]
                }
                if fanout_entry not in pre_core_neuron["fanout"]:
                    pre_core_neuron["fanout"].append(fanout_entry)

                for synapse_id in synapses:
                    synapse = synapses[synapse_id]
                    if "wt" in synapse:
                        post_core_axons[axon_dict[pre_neuron][post_neuron_core]].append({
                            "neuron": post_neuron_id,
                            "wt": synapse["wt"]
                        })
                    elif "fixed-wt" in synapse:
                        post_core_axons[axon_dict[pre_neuron][post_neuron_core]].append({
                            "neuron": post_neuron_id,
                            "fixed-wt": synapse["fixed-wt"]
                        })
            else:
                for synapse_id in synapses:
                    synapse = synapses[synapse_id]
                    post_core_axons = cores_list[post_neuron_core[0]][post_neuron_core[1]][post_neuron_core[2]]["input_axon"]
                    post_core_axons.append([{
                        "neuron": post_neuron_id,
                        "fixed-wt": synapse["fixed-wt"]
                    }])
                    synapse["axon_id"] = len(post_core_axons) - 1


def build_network_act(network, network_fanin, cores_list, cluster_size_x, cluster_size_y, core_size_x=2, core_size_y=2):
    network_act = {"ticks": network["ticks"], "cores": []}

    for i in range(cluster_size_x):
        for j in range(cluster_size_y):
            for k in range(core_size_x * core_size_y):
                network_act["cores"].append(cores_list[i][j][k])

    if "spikes_dict" in network:
        network_act["spikes"] = []
        for tick in network["spikes_dict"]:
            tick_payload = {"tick": int(tick), "cores": []}
            for syn in network["spikes_dict"][tick]["syn"]:
                neuron_core_id = network_fanin["neuron_dict"][str(syn[0])]["core"]
                tick_payload["cores"].append({
                    "x": neuron_core_id[0],
                    "y": neuron_core_id[1],
                    "id": neuron_core_id[2],
                    "input_axon": [network["synapse_dict"][str(syn[0])][str(syn[1])][str(syn[2])]["axon_id"]]
                })
            network_act["spikes"].append(tick_payload)

    return network_act


def compute_network_stats(network_act):
    stats = {
        "max_num_neuron": 0,
        "max_neuron_fanout": 0,
        "max_size_fanout_table": 0,
        "max_num_axon": 0,
        "max_num_neuron_model": 0,
        "max_num_remote": 0,
        "max_size_axon_table": 0,
        "max_num_fanin": 0
    }

    for core in network_act["cores"]:
        neuron_fanout = 0
        size_fanout_table = 0
        size_axon_table = 0

        num_neuron = len(core["neurons"])
        num_axon = len(core["input_axon"])
        num_remote = len(core["remote"])
        num_neuron_model = len(core["neuron_models"])

        for axon in core["input_axon"]:
            size_axon_table += len(axon)
        size_axon_table += num_axon

        for neuron in core["neurons"]:
            size_fanout_table += len(neuron["fanout"])
            neuron_fanout = max(neuron_fanout, len(neuron["fanout"]))
        size_fanout_table += num_neuron

        stats["max_num_neuron"] = max(stats["max_num_neuron"], num_neuron)
        stats["max_neuron_fanout"] = max(stats["max_neuron_fanout"], neuron_fanout)
        stats["max_size_fanout_table"] = max(stats["max_size_fanout_table"], size_fanout_table)
        stats["max_num_axon"] = max(stats["max_num_axon"], num_axon)
        stats["max_num_neuron_model"] = max(stats["max_num_neuron_model"], num_neuron_model)
        stats["max_num_remote"] = max(stats["max_num_remote"], num_remote)
        stats["max_size_axon_table"] = max(stats["max_size_axon_table"], size_axon_table)

    for core in network_act["cores"]:
        num_fanin = 0
        for fanin_coreinfo in network_act["cores"]:
            if {"x": core["x"], "y": core["y"], "id": core["id"]} in fanin_coreinfo["remote"]:
                num_fanin += 1
        stats["max_num_fanin"] = max(stats["max_num_fanin"], num_fanin)

    return stats



def generate_act_network_assets(
    core_mappings=None,
    network_path='network.json',
    mapping_file='core_mappings.json',
    fanin_file='network_fanin.json',
    act_file='act_network.json',
    fanin_updated_file='network_fanin_updated.json',
    num_cores=4,
    seed=42,
    balanced=True,
    core_size_x=2,
    core_size_y=2,
    save_to_nscale=True,
):
    network = load_network(network_path)


    network_fanin = build_network_fanin(network, core_mappings)
    write_json(fanin_file, network_fanin, save_to_nscale=save_to_nscale)

    cluster_size_x, cluster_size_y = get_cluster_sizes(network_fanin)

    cores_list, cores_list_neuron_model, cores_list_remote = initialize_core_data(
        cluster_size_x,
        cluster_size_y,
        core_size_x=core_size_x,
        core_size_y=core_size_y
    )

    axon_dict = register_neurons(
        network,
        network_fanin,
        cores_list,
        cores_list_neuron_model
    )

    register_synapses(
        network,
        network_fanin,
        cores_list,
        cores_list_remote,
        axon_dict
    )

    network_act = build_network_act(
        network,
        network_fanin,
        cores_list,
        cluster_size_x,
        cluster_size_y,
        core_size_x=core_size_x,
        core_size_y=core_size_y
    )

    stats = compute_network_stats(network_act)

    write_json(act_file, network_act, save_to_nscale=save_to_nscale)
    write_json(fanin_updated_file, network_fanin, save_to_nscale=save_to_nscale)

    print("successfully generate act_json/act_network.json")

    return {
        "network": network,
        "core_mappings": core_mappings,
        "network_fanin": network_fanin,
        "network_act": network_act,
        "stats": stats,
        "cluster_size_x": cluster_size_x,
        "cluster_size_y": cluster_size_y
    }


def fugu_spikes_to_json(num_neurons, matrix):
    if hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()

    spike_times = {i: [] for i in range(num_neurons)}
    for row in matrix:
        spike_times[row[1]].append(int(row[0]))

    return spike_times


if __name__ == "__main__":
    generate_act_network_assets()
