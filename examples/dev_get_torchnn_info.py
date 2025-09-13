import numpy as np
from fugu import Scaffold, scaffold
from fugu.bricks.input_bricks import Vector_Input
from fugu.bricks.dense_bricks import dense_layer_1d
from fugu.backends import snn_Backend
import numpy as np


def group_torch_layers(model):
    """
    Groups nn.Linear and snn.Leaky layers for Fugu conversion.
    Returns a list of dicts: each dict contains output_size, weights, biases, threshold, decay, and output status.
    """
    layers = []
    modules = list(model.children())
    for i in range(0, len(modules), 2):
        linear = modules[i]
        leaky = modules[i+1]
        layer_info = {
            "output_size": linear.out_features,
            "weights": linear.weight.detach().cpu().numpy(),
            "biases": linear.bias.detach().cpu().numpy(),
            "threshold": getattr(leaky, "threshold", 1.0),
            "decay": getattr(leaky, "beta", 1.0),
            "is_output": (i+2 >= len(modules))
        }
        layers.append(layer_info)
    return layers

def build_fugu_network(layer_dicts, n_steps=10):
    """
    Builds a Fugu network from grouped layer dicts.
    """
    from fugu import Scaffold
    from fugu.bricks.input_bricks import Vector_Input
    from fugu.bricks.dense_bricks import dense_layer_1d
    import numpy as np

    scaffold = Scaffold()
    # Input layer
    input_size = layer_dicts[0]["weights"].shape[1]
    spikes = np.zeros((input_size, n_steps))
    scaffold.add_brick(
        Vector_Input(spikes, coding='Raster', name='Input', time_dimension=True),
        input_nodes='input'
    )
    # Hidden/output layers
    for idx, layer in enumerate(layer_dicts):
        output_size = layer["output_size"]
        weights = layer["weights"]
        # Ensure thresholds is an array of shape (output_size,)
        threshold = layer["threshold"]
        if np.isscalar(threshold):
            threshold = np.ones(output_size) * threshold
        else:
            threshold = np.array(threshold)
            if threshold.shape != (output_size,):
                threshold = np.ones(output_size) * threshold
        brick = dense_layer_1d(
            output_shape=(output_size,),
            weights=weights,
            thresholds=threshold,
            name=f"Layer{idx}"
        )
        scaffold.add_brick(
            brick,
            input_nodes=[-1],
            output=layer["is_output"]
        )
    scaffold.lay_bricks()
    return scaffold
