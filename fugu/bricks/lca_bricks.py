from fugu.bricks import Brick
from fugu.scaffold.port import PortSpec, ChannelSpec
import numpy as np

class LCABrick(Brick):
    def __init__(self, Phi, **kwargs):
        super().__init__(**kwargs)
        self.Phi = Phi
        self.supported_codings = ['Raster', 'Undefined']

    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        """LCABrick doesn't require inputs - it generates sparse codes."""
        return {}

    @classmethod  
    def output_ports(cls) -> dict[str, PortSpec]:
        """LCABrick outputs sparse activation codes."""
        port = PortSpec(name='output')
        port.channels['data'] = ChannelSpec(name='data', coding=['Raster'])
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    def normalize_columns(self, A):
        """ Normalize columns of A to unit norm. """
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        return A / norms
                

    def build2(self, graph, inputs: dict = {}):
        from ..scaffold.port import PortData, ChannelSpec, PortSpec
        
        # Normalize dictionary columns
        Phi = self.normalize_columns(self.Phi)
        N = Phi.shape[1]  # Number of dictionary elements

        # Create complete control node
        complete_node_name = self.generate_neuron_name('complete')
        graph.add_node(complete_node_name,
                       index=-1,
                       threshold=0.0,
                       decay=0.0,
                       p=1.0,
                       potential=0.0)

        # Create LCA neurons
        neuron_names = []
        for i in range(N):
            neuron_name = self.generate_neuron_name(f"neuron_{i}")
            neuron_names.append(neuron_name)
            graph.add_node(
                neuron_name,
                index=i,
                threshold=1.0,
                potential=0.0,
                p=1.0,
            )

        # Build inhibitory connections (W = ΦᵀΦ - I)
        W = Phi.T @ Phi
        np.fill_diagonal(W, 0.0)  # Remove self-connections

        # Add lateral inhibition edges
        for i in range(N):
            for j in range(N):
                if i != j:  
                    graph.add_edge(neuron_names[i], neuron_names[j], 
                                 weight=0, delay=1)

        # Create output port data using the proper pattern
        from ..scaffold.port import PortUtil, ChannelData
        result = PortUtil.make_ports_from_specs(LCABrick.output_ports())
        output_port = result['output']
        
        # Set the data channel neurons
        data_channel = output_port.channels['data']
        data_channel.neurons = neuron_names
        
        # Set the complete channel neurons  
        complete_channel = output_port.channels['complete']
        complete_channel.neurons = [complete_node_name]
        
        return result
