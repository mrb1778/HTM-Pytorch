import torch
from torch import nn as nn


# **Spatial Pooler (`SpatialPooler`)**
# - **SDRs:** It converts the input, in this case, the one-hot representation of a character, into a stable and sparse SDR.
# - **Synapses:** Each column maintains a set of potential synapses connecting it to a subset of the input space.
# - **Permanence values:** Synapses have permanence values, representing the strength of the connection. If permanence is above a threshold, the synapse is "connected".
# - **Overlap:** The overlap score for a column is the number of its connected synapses that are on when an input is presented.
# - **Inhibition:** Only the top percentage of columns with the highest overlap are activated, ensuring a fixed sparsity level.
# - **Boosting:** Columns that are not frequently active get a "boost" to their overlap score, encouraging them to participate in learning.
#
# **Boosting mechanism details**
# The boosting logic is implemented within the `SpatialPooler`'s `_update_boost_factors` method.
# 1.  **`active_duty_cycle`:** A running average of how often each column is active is maintained.
# 2.  **`boost_factors`:** Columns with a low `active_duty_cycle` will have their `boost_factor` increased. This factor is then multiplied with the column's overlap score during processing, giving less active columns a higher chance of being selected.
# 3.  **Homeostasis:** This mechanism prevents a small number of columns from dominating all representations and ensures that the entire capacity of the model is utilized.


class SpatialPooler(nn.Module):
    permanences: torch.Tensor
    potential_synapses: torch.Tensor
    boost_factors: torch.Tensor
    active_duty_cycle: torch.Tensor
    connected_mask: torch.Tensor
    active_columns: torch.Tensor
    active_synapse_mask: torch.Tensor
    top_k_values: torch.Tensor
    top_k_indices: torch.Tensor

    def __init__(
            self,
            column_count: int,
            potential_pct: float,
            output_size: int,
            permanence_inc: float,
            permanence_dec: float,
            device: torch.device | str | None = None
    ) -> None:
        super(SpatialPooler, self).__init__()
        self.column_count = column_count
        self.potential_pool_size = int(column_count * potential_pct)
        self.output_size = output_size

        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.device = device

        # Initialize permanences for each column's potential synapses
        self.register_buffer('permanences', torch.zeros(column_count, self.potential_pool_size, device=device))
        self.register_buffer('potential_synapses',
                             torch.zeros(column_count, self.potential_pool_size, dtype=torch.long, device=device))
        self.register_buffer('boost_factors', torch.ones(column_count, device=device))
        self.register_buffer('active_duty_cycle', torch.zeros(column_count, device=device))

        self.iterations = 0
        self.learn = True

        # Connect columns to random inputs
        for c in range(column_count):
            self.potential_synapses[c] = torch.randperm(column_count, device=device)[:self.potential_pool_size]
            self.permanences[c] = 0.25 + 0.5 * torch.rand(self.potential_pool_size, device=device)

        self.register_buffer('connected_mask', (self.permanences > 0.5))

        self.register_buffer('active_columns', torch.zeros(
            self.column_count,
            device=self.device,
            dtype=torch.bool
        ), persistent=False)

        self.register_buffer('top_k_values', torch.empty(
            self.output_size,
            dtype=torch.float,
            device=device
        ), persistent=False)
        self.register_buffer('top_k_indices', torch.zeros(
            self.output_size,
            device=self.device,
            dtype=torch.long
        ), persistent=False)

        self.register_buffer('active_synapse_mask', torch.empty(
            (self.output_size, self.potential_pool_size),
            device=self.device,
            dtype=torch.bool
        ), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_to_topk(x)
        self._forward_to_active()

        if self.learn:
            self.iterations += 1
            self._learn(x)

        return self.active_columns

    def _forward_to_topk(self, x: torch.Tensor) -> None:
        synapse_inputs = x[self.potential_synapses].float()
        overlaps = torch.sum(synapse_inputs * self.connected_mask, dim=1)
        overlaps.mul_(self.boost_factors)
        torch.topk(overlaps,
                   k=self.output_size,
                   out=(self.top_k_values, self.top_k_indices))

    def _forward_to_active(self) -> None:
        self.active_columns.zero_()
        self.active_columns[self.top_k_indices] = 1

    def _learn(self, x: torch.Tensor) -> None:
        self._update_permanences(x, self.top_k_indices)
        self._update_boost_factors()

    def _update_permanences(self, x: torch.Tensor, active_column_indices) -> None:
        # Get the subset of potential synapses for the active columns
        potential_synapses_for_active = self.potential_synapses[active_column_indices]

        # Get the corresponding permanence values
        permanences_for_active = self.permanences[active_column_indices]

        # Create a mask of active synapses for the active columns
        torch.gather(x.expand(len(active_column_indices), -1),
                     dim=1,
                     index=potential_synapses_for_active,
                     out=self.active_synapse_mask)

        # Update permanences for active inputs
        permanences_for_active[self.active_synapse_mask] += self.permanence_inc

        # Update permanences for inactive inputs
        permanences_for_active[~self.active_synapse_mask] -= self.permanence_dec

        # Clamp permanences
        permanences_for_active.clamp_(0.0, 1.0)

        torch.greater(self.permanences,
                      0.5,
                      out=self.connected_mask)

    def _update_boost_factors(self) -> None:
        # Update decay rate over time
        current_decay = 0.01 / (1 + self.iterations / 1000)
        self.active_duty_cycle = (1 - current_decay) * self.active_duty_cycle + current_decay * self.active_columns

        # Update boost coefficient over time
        current_boost_coef = 10.0 / (1 + self.iterations / 1000)
        average_duty_cycle = self.active_duty_cycle.mean()
        torch.exp(
            (average_duty_cycle - self.active_duty_cycle) * current_boost_coef,
            out=self.boost_factors
        )
        self.boost_factors.clamp_(1.0, 10.0)
