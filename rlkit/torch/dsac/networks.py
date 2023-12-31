"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.torch.dsac.FeedforwardDNN import FeedForwardDNN
from rlkit.torch.dsac.MonotonicNN import MonotonicNN
import torch.nn.init as init

from rlkit.torch import pytorch_util as ptu


def softmax(x):
    return F.softmax(x, dim=-1)


class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size  = 64,
            num_quantiles   = 32,
            layer_norm=True,
            **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            linear_layer = nn.Linear(last_size, next_size)
            init.xavier_uniform_(linear_layer.weight, gain=nn.init.calculate_gain('relu'))
            init.uniform_(linear_layer.bias, -1, 1)  # Initialize bias as desired
            self.base_fc += [
                linear_layer,
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = ptu.from_numpy(np.arange(1, 1 + self.embedding_size))

    def forward(self, state, action, tau):

        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """

        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        #output = output.sort(dim=-1)[0]
        return output


class UMNN(nn.Module):

    """
    VARIABLES:  - stateEmbeddingDNN: State embedding part of the Deep Neural Network.
                - UMNN: UMNN part of the Deep Neural Network.

    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 structureDNN,
                 structureUMNN    = [128],
                 embedding_size   = 64,
                 number_of_steps  = 25,
                 num_quantiles    = 32,
                 layer_norm       = True,
                 device='cuda:0'):
        """
        GOAL: Defining and initializing the Deep Neural Network.

        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - structureDNN: Structure of the feedforward DNN for state embedding.
                - structureUMNN: Structure of the UMNN for distribution representation.
                - stateEmbedding: Dimension of the state embedding.
                - numberOfSteps: Number of integration steps for the UMNN.
                - device: Hardware device (CPU or GPU).
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(UMNN, self).__init__()

        # Initialization of the Deep Neural Network
        self.stateEmbeddingDNN = FeedForwardDNN(input_size, embedding_size, structureDNN)
        self.stateActionEmbeddingDNN = FeedForwardDNN(input_size, embedding_size, structureDNN)
        self.UMNN_net = MonotonicNN(embedding_size + 1, structureUMNN, number_of_steps, output_size, device)
        self.num_quantiles = num_quantiles

        last_size = input_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )

        self.const_vec = ptu.from_numpy(np.arange(1, 1 + embedding_size))


    # Adjustment 1
    def forward(self, state, action, taus):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.

        INPUTS: - state: RL state.
                - action: RL action
                - taus: Samples of taus.

        OUTPUTS: - output: Output of the Deep Neural Network.
        """

        torch.cuda.empty_cache()

        # State embedding part of the Deep Neural Network
        batchSize = state.size(0)

        # x = self.stateActionEmbeddingDNN(state)
        # x = x.repeat(1, int(len(taus) / len(state))).view(-1, x.size(1))

        # Expand the state and action tensors
        # state = state.repeat(1, int(len(taus) / len(state))).view(-1, state.size(1))
        # action = action.repeat(1, int(len(taus) / len(state))).view(-1, action.size(1))

        # Concatenate the state and action tensors
        x = torch.cat([state, action], dim=1)
        x = self.stateActionEmbeddingDNN(x)
        taus = taus.view(-1, 1)
        x = x.repeat(1, int(len(taus) / len(x))).view(-1, x.size(1))

        # UMNN part of the Deep Neural Network
        x = self.UMNN_net(taus, x)

        # Appropriate format
        output = torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)

        return output


class UMNN2(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size      = 64,
            num_quantiles       = 32,
            structureUMNN       = [128],
            number_of_steps_MNN = 40,           # Integration steps
            layer_norm          = True,
            embedding_sizeNN    = 32,
            device              = 'cuda:0',
            **kwargs
    ):

        super().__init__()

        self.layer_norm         = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size

        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size

        self.base_fc        = nn.Sequential(*self.base_fc)
        self.num_quantiles  = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc         = nn.Sequential(
                                nn.Linear(embedding_size, last_size),
                                nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
                                nn.Sigmoid(),
        )
        self.merge_fc       = nn.Sequential(
                                nn.Linear(last_size, hidden_sizes[-1]),
                                nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
        )
        self.last_fc    = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec  = ptu.from_numpy(np.arange(1, 1 + self.embedding_size))
        self.UMNN_net   = MonotonicNN(embedding_sizeNN + 1, structureUMNN, number_of_steps_MNN, output_size, device)

    def forward(self, state, action, tau):

        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """
        batchSize = state.size(0)
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        x = self.last_fc(h).squeeze(-1)  # (N, T)
        tau = tau.view(-1, 1)
        x = x.repeat(1, int(len(tau) / len(x))).view(-1, x.size(1))
        x = self.UMNN_net(tau, x)
        output = torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)
        return output

class UMNN3(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            structureUMNN               = [64],
            embedding_sizeNN            = 128,
            number_of_steps_MNN         = 40,
            device                      = 'cuda:0'
    ):
        super(UMNN3, self).__init__()

        # Initialization of the Deep Neural Network
        self.stateEmbeddingDNN = FeedForwardDNN(input_size, embedding_sizeNN, hidden_sizes)
        self.UMNN              = MonotonicNN(embedding_sizeNN+1, structureUMNN, number_of_steps_MNN, output_size, device)

    def forward(self, state, actions, taus):

        # State and action embedding part of the DNN
        batchSize   = state.size(0)
        h           = torch.cat([state, actions], dim=1)
        x           = self.stateEmbeddingDNN(h)

        # UMNN part
        taus        = taus.view(-1, 1)
        x           = x.repeat(1, int(len(taus)/len(h))).view(-1, x.size(1))

        x           = self.UMNN(taus, x)
        output      = torch.cat(torch.chunk(torch.transpose(x, 0, 1), batchSize, dim=1), 0)

        return output

