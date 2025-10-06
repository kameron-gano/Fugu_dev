from fugu.bricks import Brick
import numpy as np

class LCABrick(Brick):
    def __init__(self, Phi, **kwargs):
        super().__init__(**kwargs)
        self.Phi = Phi

    def normalize_columns(A):
        """ Normalize columns of A to unit norm. """
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        return A / norms
                

    def build2(self, graph):

        new_begin = f"LCA_begin"
        new_complete = f"LCA_complete"

        # TODO: ADD IN EDGES FOR CONTROL NODES 
        # TODO: MAKE SURE RETURN VALUES ARE CORRECT AND COMPLETE
        
        # Begin node: fires when upstream begin fires (delay 1)
        graph.add_node(new_begin, index=-2, threshold=0.0, decay=0.0, p=1.0, potential=0.0, layer='control')
        # Complete node: fires after both upstream completes (OR/sum behavior)
        graph.add_node(new_complete, index=-1, threshold=0.9, decay=0.0, p=1.0, potential=0.0, layer='control')


        # Normalize dictionary columns
        Phi = self.normalize_columns(self.Phi)
        N = Phi.shape[1]

        # Create neurons
        for i in range(N):
            name = f"neuron_{i}"
            graph.add_node(
                name,
                threshold=1.0,
                potential=0.0,
                p=1.0,
            )

        # Build connections (W = ΦᵀΦ)
        W = Phi.T @ Phi
        np.fill_diagonal(W, 0.0)

        # add edges to graph
        for i in range(N):
            for j in range(N):
                if i != j:
                    graph.add_edge(f"neuron_{j}", f"neuron_{i}", weight=-W[i, j])

        return graph
