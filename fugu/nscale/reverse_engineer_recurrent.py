"""Reverse-engineer an ACT-formatted network back into a Fugu graph, run simulation,
and compare generated spikes with a reference spikes JSON.

Usage (from repo root):
    python fugu/nscale/reverse_engineer_recurrent.py \
      --act-network fugu/nscale_old/act_network.json \
      --fanin-updated fugu/nscale_old/network_fanin_updated.json \
      --output fugu/nscale_old/spikes_reverse_fugu.json \
      --reference fugu/nscale_old/spikes_fugu.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fugu
from fugu import Brick, Scaffold
from fugu.backends import snn_Backend


CoreKey = Tuple[int, int, int]
NeuronKey = Tuple[CoreKey, int]


class ReconstructedNetworkBrick(Brick):
    def __init__(self, neuron_params: Dict[int, dict], edges: Dict[Tuple[int, int], float], name: str = "reconstructed"):
        super().__init__()
        self.neuron_params = neuron_params
        self.edges = edges
        self.name = name
        self.is_built = False
        self.metadata = {"D": 1}
        self.supported_codings = fugu.input_coding_types

    def build(self, graph, metadata, control_nodes, input_lists, input_codings):
        output_codings: List[str] = []

        for neuron_id in sorted(self.neuron_params):
            params = self.neuron_params[neuron_id]
            node_name = f"{self.name}_{neuron_id}"
            graph.add_node(
                node_name,
                index=neuron_id,
                threshold=float(params["threshold"]),
                decay=float(params["decay"]),
                bias=float(params["bias"]),
                p=float(params["p"]),
                potential=0.0,
            )

        for (pre_id, post_id), weight in sorted(self.edges.items()):
            graph.add_edge(
                f"{self.name}_{pre_id}",
                f"{self.name}_{post_id}",
                weight=float(weight),
                delay=1.0,
            )

        self.is_built = True
        output_lists: List = []
        return graph, self.metadata, [], output_lists, output_codings


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _build_global_id_lookup(
    fanin_updated: Optional[dict],
    local_neuron_order: Iterable[NeuronKey],
) -> Dict[NeuronKey, int]:
    if fanin_updated is not None:
        lookup: Dict[NeuronKey, int] = {}
        for neuron_id_str, info in fanin_updated["neuron_dict"].items():
            core = tuple(info["core"])
            local_idx = int(info["neuron index"])
            lookup[(core, local_idx)] = int(neuron_id_str)

        missing = [key for key in local_neuron_order if key not in lookup]
        if missing:
            raise ValueError(
                "Missing core/index mapping entries in fanin_updated for "
                f"{len(missing)} neurons. Example missing key: {missing[0]}"
            )
        return lookup

    return {key: i for i, key in enumerate(local_neuron_order)}


def reverse_engineer_from_act(act_network: dict, fanin_updated: Optional[dict] = None):
    cores = act_network["cores"]

    core_lookup: Dict[CoreKey, dict] = {
        (int(core["x"]), int(core["y"]), int(core["id"])): core for core in cores
    }

    local_neuron_order: List[NeuronKey] = []
    for core_key in sorted(core_lookup):
        core = core_lookup[core_key]
        for local_idx, _ in enumerate(core["neurons"]):
            local_neuron_order.append((core_key, local_idx))

    global_id_lookup = _build_global_id_lookup(fanin_updated, local_neuron_order)

    neuron_params: Dict[int, dict] = {}
    for core_key, local_idx in local_neuron_order:
        core = core_lookup[core_key]
        neuron_def = core["neurons"][local_idx]
        model_idx = int(neuron_def["model"]) - 1
        model = core["neuron_models"][model_idx]
        global_id = global_id_lookup[(core_key, local_idx)]

        neuron_params[global_id] = {
            "threshold": model["th"],
            "decay": model.get("leak", 0.0),
            "bias": model.get("bias", 0.0),
            "p": 1.0,
        }

    edge_weights: Dict[Tuple[int, int], float] = defaultdict(float)
    for pre_core_key in sorted(core_lookup):
        pre_core = core_lookup[pre_core_key]
        for pre_local_idx, pre_neuron in enumerate(pre_core["neurons"]):
            pre_global = global_id_lookup[(pre_core_key, pre_local_idx)]

            for fanout in pre_neuron.get("fanout", []):
                remote_idx = int(fanout["remote"])
                axon_idx = int(fanout["axon"])

                remote_core_ref = pre_core["remote"][remote_idx]
                post_core_key = (
                    int(remote_core_ref["x"]),
                    int(remote_core_ref["y"]),
                    int(remote_core_ref["id"]),
                )
                post_core = core_lookup[post_core_key]

                syn_entries = post_core["input_axon"][axon_idx]
                for syn in syn_entries:
                    post_local_idx = int(syn["neuron"])
                    post_global = global_id_lookup[(post_core_key, post_local_idx)]
                    weight = float(syn.get("fixed-wt", syn.get("wt", 1.0)))
                    edge_weights[(pre_global, post_global)] += weight

    return neuron_params, dict(edge_weights)


def run_fugu_behavior(neuron_params: Dict[int, dict], edge_weights: Dict[Tuple[int, int], float], ticks: int) -> Dict[str, List[int]]:
    scaffold = Scaffold()
    scaffold.add_brick(
        ReconstructedNetworkBrick(neuron_params=neuron_params, edges=edge_weights),
        [],
        output=True,
    )
    scaffold.lay_bricks()

    id_by_neuron_number: Dict[int, int] = {}
    for node_name, attrs in scaffold.graph.nodes(data=True):
        neuron_number = int(attrs["neuron_number"])
        original_id = int(node_name.rsplit("_", 1)[1])
        id_by_neuron_number[neuron_number] = original_id

    backend = snn_Backend()
    backend.compile(scaffold, {"record": "all"})
    result = backend.run(ticks)

    spikes: Dict[str, List[int]] = {str(neuron_id): [] for neuron_id in sorted(neuron_params)}
    matrix = result.to_numpy()
    for row in matrix:
        tick = int(row[0])
        neuron_number = int(row[1])
        original_id = id_by_neuron_number[neuron_number]
        spikes[str(original_id)].append(tick)

    return spikes


def normalize_spikes(spikes: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {
        str(k): sorted(int(t) for t in v)
        for k, v in sorted(spikes.items(), key=lambda kv: int(kv[0]))
    }


def compare_spikes(candidate: Dict[str, List[int]], reference: Dict[str, List[int]]) -> Tuple[bool, List[str]]:
    cand = normalize_spikes(candidate)
    ref = normalize_spikes(reference)

    if cand == ref:
        return True, []

    details: List[str] = []
    all_keys = sorted(set(cand) | set(ref), key=int)
    for neuron_id in all_keys:
        c = cand.get(neuron_id)
        r = ref.get(neuron_id)
        if c != r:
            details.append(f"Neuron {neuron_id}: generated={c} reference={r}")
        if len(details) >= 20:
            details.append("... truncated after 20 mismatches")
            break

    return False, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-engineer ACT network and re-run Fugu behavior.")
    parser.add_argument("--act-network", type=Path, required=True)
    parser.add_argument("--fanin-updated", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--ticks", type=int, default=None)
    args = parser.parse_args()

    act_network = _load_json(args.act_network)
    fanin_updated = _load_json(args.fanin_updated) if args.fanin_updated else None

    neuron_params, edge_weights = reverse_engineer_from_act(act_network, fanin_updated)
    ticks = int(args.ticks if args.ticks is not None else act_network["ticks"])

    spikes = run_fugu_behavior(neuron_params, edge_weights, ticks=ticks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(normalize_spikes(spikes), f)

    print(f"Wrote reverse-engineered spikes to: {args.output}")

    if args.reference and args.reference.exists():
        reference = _load_json(args.reference)
        is_match, details = compare_spikes(spikes, reference)
        print(f"Matches reference: {is_match}")
        if not is_match:
            for line in details:
                print(line)


if __name__ == "__main__":
    main()
