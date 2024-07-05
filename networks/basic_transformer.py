# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for a basic transformer model
import torch.nn as nn
from globals import *

class BasicTransformer(nn.Module):
    """
    Torch nn module for a basic model containing 2 linear layers around a transformer (time dimension).
    """
    def __init__(self,
                num_inputs,
                linear_1_size=BASIC_TRANSFORMER_LINEAR_1_SIZE,
                transformer_dim=BASIC_TRANSFORMER_DIM,
                num_heads=BASIC_TRANSFORMER_NUM_HEADS,
                num_layers=BASIC_TRANSFORMER_NUM_LAYERS,
                num_outputs=BASIC_TRANSFORMER_LINEAR_2_SIZE):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        num_inputs : int
            size of input features
        linear_1_size : int
            size of 1st dense layer, and input/output of transformer
        transformer_dim : int
            dimension of the transformer model
        num_heads : int
            number of attention heads in the transformer
        num_layers : int
            number of transformer layers
        linear_2_size : int
            size of 2nd dense layer, and number of outputs
        """
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # create layers
        self.linear1 = nn.Linear(num_inputs, linear_1_size)
        self.leakyReLU1 = nn.LeakyReLU()

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=linear_1_size,
            nhead=num_heads,
            dim_feedforward=transformer_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.linear2 = nn.Linear(linear_1_size, num_outputs)
        self.leakyReLU2 = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(num_outputs)

    def forward(self, inputs):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        inputs : tensor (batch, frames, input_size)
            input data to be fed thru model
  
        Returns
        ----------
        outputs : tensor (batch, frames, output_size)
            output data from model
        """

        # feed inputs through
        linear1_out = self.linear1(inputs)
        leaky1_out = self.leakyReLU1(linear1_out)

        # apply transformer
        transformer_out = self.transformer(leaky1_out)

        linear2_out = self.linear2(transformer_out)
        leaky2_out = self.leakyReLU2(linear2_out)
        outputs = self.layer_norm(leaky2_out)

        return outputs

