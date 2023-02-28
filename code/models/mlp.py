import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layers=[3, 128, 128, 3], nonlinearity=nn.SiLU, use_gn=True, skip_input_idx=None):
        '''
        If skip_input_idx is not None, the input feature after idx skip_input_idx will be skip connected to every later of the MLP.
        '''
        super(MLP, self).__init__()

        in_size = layers[0]
        out_channels = layers[1:]

        # input layer
        layers = []
        layers.append(nn.Linear(in_size, out_channels[0]))
        skip_size = 0 if skip_input_idx is None else (in_size - skip_input_idx)
        # now the rest
        for layer_idx in range(1, len(out_channels)):
            fc_layer = nn.Linear(out_channels[layer_idx-1], out_channels[layer_idx])
            if use_gn:
                bn_layer = nn.GroupNorm(16, out_channels[layer_idx-1])
                layers.append(bn_layer)
            layers.extend([nonlinearity(), fc_layer])
        self.net = nn.ModuleList(layers)
        self.skip_input_idx = skip_input_idx

    def forward(self, x):
        '''
        B x D x * : batch norm done over dim D
        '''
        D = x.shape[-1]
        B = x.shape[0]
        x = x.reshape(-1, D)
        skip_in = None
        if self.skip_input_idx is not None:
            skip_in = x[:,self.skip_input_idx:]
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i==0 and self.skip_input_idx is not None:
                skip_in = x[:,self.skip_input_idx:]
            elif self.skip_input_idx is not None and i < len(self.net)-1 and isinstance(layer, nn.Linear):
                x = x+skip_in
        x = x.reshape(B, -1)
        return x