import torch
import torch.nn as nn
from torch import optim
from UMNNModel import UMNN_Model

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, embedding_size, num_quantiles, num_steps, device):
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.device = device

        self.umnn = UMNN_Model(
            numberOfInputs  = state_dim + action_dim,
            numberOfOutputs = num_quantiles,
            structureDNN    = [128, 128],
            structureUMNN   = [128, 128],
            stateEmbedding  = embedding_size,
            numberOfSteps   = num_steps,
            device          = device
        ).to(device)

        self.critic_optimizer = optim.Adam(self.umnn.parameters(), lr=3e-4)

    def forward(self, state, action, taus):
        state_action = torch.cat([state, action], dim=-1)
        quantile_values = self.umnn(state_action, taus)
        return quantile_values

    def update(self, state, action, taus, target_quantile_values):
        self.critic_optimizer.zero_grad()
        quantile_values = self.forward(state, action, taus)
        loss = quantile_regression_loss(quantile_values, target_quantile_values, taus, 1.0)
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()

    def get_quantile_values(self, state, action, taus):
        quantile_values = self.forward(state, action, taus)
        return quantile_values