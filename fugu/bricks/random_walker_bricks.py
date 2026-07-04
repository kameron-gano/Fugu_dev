import numpy as np
from fugu.bricks import Brick
from fugu.scaffold.port import ChannelSpec, PortSpec, PortUtil


class RandomWalker(Brick):
    def __init__(
        self,
        transition_matrix,
        init_walkers,
        timesteps, 
        name="RandomWalker",
    ):
        super().__init__(name=name)
        self.P = np.asarray(transition_matrix, dtype=float)

        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("transition_matrix must be a square 2-D array.")
        self.N = self.P.shape[0]        
        if self.N == 0:
            raise ValueError("transition_matrix must have at least one room.")
        if np.any(self.P < 0):
            raise ValueError("transition_matrix entries must be non-negative.")

        self.timesteps = int(timesteps)
        if self.timesteps < 1:
            raise ValueError("timesteps must be >= 1.")

        self.init_walkers = {int(k): int(v) for k, v in init_walkers.items()}
        if not self.init_walkers:
            raise ValueError("init_walkers must seed at least one walker.")
        for room, count in self.init_walkers.items():
            if not (0 <= room < self.N):
                raise ValueError(f"init_walkers references room {room} outside 0..{self.N - 1}.")
            if count <= 0:
                raise ValueError(f"walker count for room {room} must be positive.")

        self.supported_codings = ['Raster', 'Undefined']

    @classmethod
    def input_ports(cls):
        """Source brick: takes no spike inputs (problem data comes via __init__)."""
        return {}

    @classmethod
    def output_ports(cls):
        port = PortSpec(name='output')
        port.channels['data'] = ChannelSpec(name='data', coding=['Raster'])
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    def build2(self, graph, inputs={}):
        rooms = {}
        for i in range(self.N):
            neighbours = self._row_to_neighbours(i)
            rooms[i] = self._add_room(graph, i, neighbours)

        ctrl = self._add_controllers(graph)

        self._wire_rooms(graph, rooms, ctrl)

        self._connect_rooms(graph, rooms)

        self._seed_walkers(graph, rooms, ctrl)

        complete = self._add_completion(graph)

        readout_names = [rooms[i]['readout'] for i in range(self.N)]
        result = PortUtil.make_ports_from_specs(RandomWalker.output_ports())
        out = result['output']
        out.channels['data'].neurons = readout_names
        out.channels['complete'].neurons = [complete]
        return result

    def _add_room(self, graph, i, neighbours):
        idx = (i,)
        probability_bits = max(len(neighbours) - 1, 1)

        counter = self._room_neuron(i, 'counter')
        graph.add_node(counter, index=idx, threshold=0.5, decay=0.0, p=1.0, potential=0.0)

        generator = self._room_neuron(i, 'generator')
        graph.add_node(generator, index=idx, threshold=0.5, decay=1.0, p=1.0, potential=0.0)

        readout = self._room_neuron(i, 'readout')
        graph.add_node(readout, index=idx, threshold=0.5, decay=0.0, p=1.0, potential=0.0)

        buffer = self._room_neuron(i, 'buffer')
        graph.add_node(buffer, index=idx, threshold=0.5, decay=0.0, p=1.0, potential=0.0)

        buffer_control = self._room_neuron(i, 'buffer_control')
        graph.add_node(buffer_control, index=idx, threshold=0.5, decay=1.0, p=1.0, potential=0.0)

        probs = [p for (_, p) in neighbours] or [0.0]
        random_gates = []
        for k in range(probability_bits):
            denom = 1.0 - sum(probs[:k])
            p_k = (probs[k] / denom) if denom > 0 else 0.0
            g = self._room_neuron(i, f'gate{k}')
            graph.add_node(g, index=idx, threshold=0.5, decay=1.0, p=p_k, potential=0.0)
            random_gates.append(g)

        output_gates = []
        for k in range(probability_bits + 1):
            g = self._room_neuron(i, f'outgate{k}')
            graph.add_node(g, index=idx, threshold=0.5, decay=1.0, p=1.0, potential=0.0)
            output_gates.append(g)

        return {
            'neighbours': neighbours,
            'counter': counter,
            'generator': generator,
            'readout': readout,
            'buffer': buffer,
            'buffer_control': buffer_control,
            'random_gates': random_gates,
            'output_gates': output_gates,
        }

    def _wire_rooms(self, graph, rooms, ctrl):
        for room in rooms.values():
            counter = room['counter']
            generator = room['generator']
            readout = room['readout']
            buffer = room['buffer']
            buffer_control = room['buffer_control']
            gates = room['random_gates']
            outs = room['output_gates']
            output_length = len(outs)

            graph.add_edge(buffer_control, counter,        weight=-1.0, delay=1)
            graph.add_edge(buffer_control, buffer_control, weight=1.0,  delay=1)
            graph.add_edge(buffer_control, buffer,         weight=1.0,  delay=1)
            graph.add_edge(buffer,         counter,        weight=1.0,  delay=1)
            graph.add_edge(buffer,         buffer_control, weight=-1.0, delay=1)
            graph.add_edge(buffer,         buffer,         weight=-1.0, delay=1)
            graph.add_edge(ctrl['buffer_supervisor'], buffer,         weight=1.0, delay=1)
            graph.add_edge(ctrl['buffer_supervisor'], buffer_control, weight=1.0, delay=1)
            graph.add_edge(buffer, ctrl['buffer_clear'], weight=1.0, delay=1)

            graph.add_edge(generator, readout,   weight=1.0,  delay=1)
            graph.add_edge(generator, generator, weight=1.0,  delay=1)
            graph.add_edge(generator, counter,   weight=1.0,  delay=1)
            graph.add_edge(counter,   readout,   weight=-1.0, delay=1)
            graph.add_edge(counter,   counter,   weight=-1.0, delay=1)
            graph.add_edge(counter,   generator, weight=-1.0, delay=1)

            for k in range(output_length - 1):
                graph.add_edge(generator, gates[k], weight=1.0,  delay=k + 1)
                graph.add_edge(counter,   gates[k], weight=-1.0, delay=k + 1)
                graph.add_edge(gates[k],  outs[k],  weight=1.0,  delay=1)
                d = 1
                for j in range(k + 1, output_length - 1):
                    graph.add_edge(gates[k], gates[j], weight=-1.0, delay=d)
                    d += 1
                graph.add_edge(gates[k], outs[-1], weight=-1.0, delay=d)
            graph.add_edge(generator, outs[-1], weight=1.0,  delay=output_length)
            graph.add_edge(counter,   outs[-1], weight=-1.0, delay=output_length)

            graph.add_edge(ctrl['walker_supervisor'], counter,   weight=1.0, delay=1)
            graph.add_edge(ctrl['walker_supervisor'], generator, weight=1.0, delay=1)
            graph.add_edge(counter, ctrl['walks_complete'], weight=1.0, delay=output_length)

    def _connect_rooms(self, graph, rooms):
        for room in rooms.values():
            for k, (j, _p) in enumerate(room['neighbours']):
                gate = room['output_gates'][k]
                graph.add_edge(gate, rooms[j]['buffer'], weight=-1.0, delay=1)

    def _add_controllers(self, graph):
        g = self.generate_neuron_name
        walker_supervisor = g('walker_supervisor')
        walks_complete = g('walks_complete')
        buffer_supervisor = g('buffer_supervisor')
        buffer_clear = g('buffer_clear')

        graph.add_node(walker_supervisor, threshold=1.0, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(buffer_supervisor, threshold=1.0, decay=1.0, p=1.0, potential=0.0)
        graph.add_node(walks_complete, threshold=self.N - 0.5, decay=0.0, p=1.0, potential=0.0)
        graph.add_node(buffer_clear,   threshold=self.N - 0.5, decay=0.0, p=1.0, potential=0.0)

        graph.add_edge(walks_complete, buffer_supervisor, weight=2.0, delay=2)
        graph.add_edge(buffer_clear,   walker_supervisor, weight=2.0, delay=2)

        return {
            'walker_supervisor': walker_supervisor,
            'walks_complete': walks_complete,
            'buffer_supervisor': buffer_supervisor,
            'buffer_clear': buffer_clear,
        }

    def _seed_walkers(self, graph, rooms, ctrl):
        graph.nodes[ctrl['walker_supervisor']]['potential'] = 10.0
        for room, count in self.init_walkers.items():
            graph.nodes[rooms[room]['counter']]['potential'] = -float(count)

    def _add_completion(self, graph):
        g = self.generate_neuron_name
        complete = g('complete')
        timestep_counter = g('timestep_counter')

        graph.add_node(complete, index=(-1,), threshold=float(self.timesteps),
                       decay=0.0, p=1.0, potential=0.0)
        graph.add_node(timestep_counter, index=(-2,), threshold=0.5,
                       decay=0.0, p=1.0, potential=1.0)
        graph.add_edge(timestep_counter, timestep_counter, weight=1.0, delay=1)
        graph.add_edge(timestep_counter, complete, weight=1.0, delay=1)
        return complete

    def _room_neuron(self, i, suffix):
        return self.generate_neuron_name(f'room{i}_{suffix}')

    def _row_to_neighbours(self, i):
        return [(j, float(self.P[i, j])) for j in range(self.N) if self.P[i, j] > 0]
