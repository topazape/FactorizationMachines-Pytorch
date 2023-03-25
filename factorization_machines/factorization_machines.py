import torch
from torch import nn


class FactorizationMachines(nn.Module):
    """Factorization Machines."""

    def __init__(
        self,
        input_numerical_dim: int,
        input_categorical_dim: int,
        embedding_dim: int,
        k: int,
    ) -> None:
        """Initialize.

        Args:
            input_numerical_dim (int): Numerical features dimension.
            input_categorical_dim (int): Categorical features dimension.
            embedding_dim (int): Embedding dimension.
            k (int): Number of factors.

        References:
            [1] Steffen Rendle. Factorization machines. 2010.
        """
        super().__init__()

        # embedding layers
        self.embedding = nn.Embedding(input_categorical_dim, embedding_dim)

        # concatenate numerical and embedded categorical features dims
        n = input_numerical_dim + embedding_dim

        # linear layer
        self.linear = nn.Linear(n, 1, bias=True)
        # for factorization
        self.v = nn.Parameter(torch.randn(n, k), requires_grad=True)

    def forward(
        self, numerical_x: torch.Tensor, categorical_x: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation.

        y_hat = w_0 + Σ w_i * x_i + Σ Σ <v_i, v_j> * x_i * x_j
              = (w_0 + Σ w_i * x_i) + (0.5 * Σ Σ (v_i,f * x_i)² - 0.5 * Σ Σ (v_i,f² * x_i²))
              = linear_terms + interaction_terms.

        Args:
            numerical_x (torch.Tensor): Numerical features.
            categorical_x (torch.Tensor): Categorical features.

        Returns:
            torch.Tensor: Predicted values.
        """
        categorical_embed = self.embedding(categorical_x).sum(dim=1)
        x = torch.cat([numerical_x, categorical_embed], dim=1)

        linear_terms = self.linear(x)
        inter1 = torch.pow(torch.matmul(x, self.v), 2)
        inter2 = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))
        interaction_terms = 0.5 * torch.sum(inter1 - inter2, dim=1, keepdim=True)

        logits = linear_terms + interaction_terms

        return logits
