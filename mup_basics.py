from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam
import torch
import torch.nn as nn
import mup 

class MyModel(nn.Module):
    def __init__(self, width, indim=1, outdim=1):
        super(MyModel, self).__init__()
        ### In model definition, replace output layer with MuReadout
        # readout = nn.Linear(width, d_out)
        self.fc1 = nn.Linear(indim, width)
        self.readout = MuReadout(width, outdim)
        ### If tying weights with an input nn.Embedding layer, do
        # readout = MuSharedReadout(input_layer.weight)

    def forward(self, x):
        return self.readout(self.fc1(x))

### Instantiate a base model
base_model = MyModel(width=1)
### Optionally, use `torchdistx.deferred_init.deferred_init` to avoid instantiating the parameters
### Simply install `torchdistx` and use
# base_model = torchdistx.deferred_init.deferred_init(MyModel, width=1)
### Instantiate a "delta" model that differs from the base model
###   in all dimensions ("widths") that one wishes to scale.
### Here it's simple, but e.g., in a Transformer, you may want to scale
###   both nhead and dhead, so the delta model should differ in both.
delta_model = MyModel(width=2) # Optionally use `torchdistx` to avoid instantiating

### Instantiate the target model (the model you actually want to train).
### This should be the same as the base model except 
###   the widths could be potentially different.
### In particular, base_model and model should have the same depth.
model = MyModel(width=100)

### Set base shapes
### When `model` has same parameter shapes as `base_model`,
###   `model` behaves exactly the same as `base_model`
###   (which is in PyTorch's default parametrization).
###   This provides backward compatibility at this particular model size.
###   Otherwise, `model`'s init and LR are scaled by μP.
### IMPORTANT: this should be called as soon as possible,
###   before re-initialization and optimizer definition.
set_base_shapes(model, base_model, delta=delta_model)

### Alternatively, one can save the base model shapes in a file
# make_base_shapes(base_model, delta_model, filename)
### and later set base shapes directly from the filename
# set_base_shapes(model, filename)
### This is useful when one cannot fit both 
###   base_model and model in memory at the same time

### Replace your custom init, if any
for param in model.parameters():
    ### If initializing manually with fixed std or bounds,
    ### then replace with same function from mup.init
    # torch.nn.init.uniform_(param, -0.1, 0.1)
    mup.init.uniform_(param, -0.1, 0.1)
    ### Likewise, if using
    ###   `xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_`
    ### from `torch.nn.init`, replace with the same functions from `mup.init`

### Use the optimizers from `mup.optim` instead of `torch.optim`
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = MuSGD(model.parameters(), lr=0.1)

### Then just train normally