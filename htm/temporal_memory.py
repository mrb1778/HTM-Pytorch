import torch
from torch import nn as nn, Tensor

from xutils.dl.pytorch import utils as pyu


# **Temporal Memory (`TemporalMemory`)** -
# **Sequence Learning:**
# The TM learns sequences of active columns produced by the SP. It associates previous active cells
# with the current active cells. -
# **Prediction:**
# When a known sequence is replayed, the TM predicts which cells in a column are about to become active. -
# **Learning:**
# If a prediction is correct, the connections are strengthened. If not, new connections are formed. -
# **Bursting:**
# If the TM fails to make a prediction for an active column, all cells within that column become active.
# This allows the SP to learn a new representation, and the TM learns the sequence from this new representation.
#


class TemporalMemory(nn.Module):
    permanence_inc: torch.Tensor
    permanence_dec: torch.Tensor
    columns: torch.Tensor
    num_segments: torch.Tensor
    column_connection_counts: torch.Tensor
    segment_post_all: torch.Tensor
    segment_post: torch.Tensor
    synapses_all: torch.Tensor
    synapses: torch.Tensor
    permanences_all: torch.Tensor
    permanences: torch.Tensor
    over_threshold_segments_all: torch.Tensor
    over_threshold_segments: torch.Tensor
    under_threshold_segments_all: torch.Tensor
    under_threshold_segments: torch.Tensor
    previous_active_locations: torch.Tensor
    active_synapses: torch.Tensor
    predicted_locations: torch.Tensor
    segment_post_max: torch.Tensor

    def __init__(
            self: 'TemporalMemory',
            num_columns: int = 1024,
            cells_per_column: int = 32,
            input_size: int = 32,
            segment_size: int = 32,
            segment_threshold: int = 16,
            permanence_threshold: float = 0.5,
            permanence_inc: float = 0.1,
            permanence_dec: float = 0.01,
            max_segments: int = 5000,
            initial_permanence: float = 0.21,
            device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.debug = False
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.input_size = input_size
        self.total_cells = num_columns * cells_per_column
        self.device = device

        self.segment_size = segment_size
        self.segment_threshold = segment_threshold
        self.permanence_threshold = permanence_threshold

        self.register_buffer('permanence_inc',
                             torch.tensor(permanence_inc, device=device))
        self.register_buffer('permanence_dec',
                             torch.tensor(-permanence_dec, device=device))
        self.max_segments = max_segments
        self.initial_permanence = initial_permanence

        num_layers = 7
        self.register_buffer('columns', torch.zeros(
            size=(
                num_layers,
                self.num_columns,
                self.cells_per_column,
            ),
            dtype=torch.bool,
            device=self.device
        ), persistent=False)
        split_columns = torch.split(self.columns,
                                    split_size_or_sections=1,
                                    dim=0)
        split_squeezed_columns = (t.squeeze(0) for t in split_columns)
        (
            self.active_and_pred,
            self.active_not_pred,
            self.pred_and_burst,
            self.burst_winners,
            self.pred_and_burst_winners,
            self.under_threshold_pred,
            self.predicted
        ) = split_squeezed_columns

        self.register_buffer('num_segments',
                             torch.tensor(0, dtype=torch.long))

        self.register_buffer('column_connection_counts', torch.zeros(
            size=(
                self.num_columns,
                self.cells_per_column,
            ),
            dtype=torch.long,
            device=self.device))

        # Index of the postsynaptic cell for each segment: (max_segments)
        self.register_buffer('segment_post_all', torch.zeros(
            size=(self.max_segments,),
            dtype=torch.long,
            device=self.device
        ))
        self.register_buffer('segment_post', None)

        # Presynaptic cell indices for each synapse: (max_segments, max_synapses_per_segment)
        self.register_buffer('synapses_all', torch.zeros(
            size=(self.max_segments, self.segment_size),
            dtype=torch.long,
            device=self.device
        ))
        self.register_buffer('synapses', None)

        self.register_buffer('permanences_all', torch.zeros(
            size=(self.max_segments, self.segment_size),
            dtype=torch.float,
            device=self.device
        ))
        self.register_buffer('permanences', None)

        # temp storage for active segments
        self.register_buffer('over_threshold_segments_all', torch.zeros(
            size=(self.max_segments,),
            dtype=torch.bool,
            device=self.device
        ))
        self.register_buffer('over_threshold_segments', None)
        self.register_buffer('under_threshold_segments_all', torch.zeros(
            size=(self.max_segments,),
            dtype=torch.bool,
            device=self.device
        ))
        self.register_buffer('under_threshold_segments', None)
        self.set_segment_shortcuts()

        self.register_buffer('previous_active_locations',
                             torch.zeros(
                                 (self.input_size,),
                                 dtype=torch.long,
                                 device=self.device,
                             ),
                             persistent=False)

        self.register_buffer('active_synapses',
                             None,
                             persistent=False)
        self.register_buffer('predicted_locations',
                             None,
                             persistent=False)
        self.register_buffer('segment_post_max', torch.zeros(
            size=(self.total_cells,),
            device=self.device,
            dtype=torch.int32
        ), persistent=False)

        self.register_load_state_dict_post_hook(self._on_post_load)
        self.iteration = 0

    def _on_post_load(self, module, incompatible_keys):
        self.set_segment_shortcuts()

    def set_segment_shortcuts(self):
        self.segment_post = self.segment_post_all[:self.num_segments]
        self.synapses = self.synapses_all[:self.num_segments]
        self.permanences = self.permanences_all[:self.num_segments]
        self.over_threshold_segments = self.over_threshold_segments_all[:self.num_segments]
        self.under_threshold_segments = self.under_threshold_segments_all[:self.num_segments]

    def reset(self):
        self.columns.zero_()
        self.segment_post_max.zero_()
        self.previous_active_locations.zero_()
        self.active_synapses = None
        self.predicted_locations = None
        self.over_threshold_segments.zero_()
        self.under_threshold_segments.zero_()

    def forward(self, active_columns: torch.Tensor) -> Tensor:
        self.iteration += 1
        pyu.nonzero_flatten(self.pred_and_burst_winners,
                            out=self.previous_active_locations)

        self._activate_predicted_cells(active_columns)
        self._activate_bursting_cells()

        if self.training:
            self._learn()

        if self.iteration > 1:
            self._predict_cells()
            if self.debug:
                print("self.predicted", self.predicted.nonzero())

        return self.predicted

    def _activate_predicted_cells(self, active_columns: torch.Tensor) -> None:
        # p   a   g  >a&p apc anp>act
        # 010 011 011 010 010 001 011
        # 100 011 001 000 010 001 001
        # 010 011 011 010 010 001 011

        self.pred_and_burst_winners.zero_()

        self.active_and_pred.copy_(self.predicted & active_columns.unsqueeze(1))
        any_active_pred_1d = self.active_and_pred.sum(dim=1) > 0
        self.active_not_pred.copy_(active_columns.unsqueeze(1) & ~any_active_pred_1d.unsqueeze(1))

        torch.bitwise_or(
            self.active_not_pred,
            self.active_and_pred,
            out=self.pred_and_burst
        )
        if self.debug:
            print("active not pred", self.active_not_pred.sum(dim=1).nonzero().numel())
            print("active and pred", self.active_and_pred.sum(dim=1).nonzero().numel())

    def _activate_bursting_cells(self) -> None:
        self._set_winner_bursting_cells()
        torch.bitwise_or(
            self.active_and_pred,
            self.burst_winners,
            out=self.pred_and_burst_winners
        )

    def _set_winner_bursting_cells(self):
        self.burst_winners.zero_()
        self._pick_active_burst_cells()

        active_rows = self.active_not_pred.sum(dim=1) > 0
        winner_rows = self.burst_winners.sum(dim=1) > 0
        remaining_cols = (active_rows & ~winner_rows)

        if remaining_cols.any():
            remaining_cols_loc = torch.nonzero(remaining_cols).view(-1)
            count_cols = self.column_connection_counts[remaining_cols_loc]

            min_counts, _ = count_cols.min(dim=1, keepdim=True)

            is_min_mask = (count_cols == min_counts)

            # 1. Assign a small random value to every cell to act as a tie-breaker
            random_noise = torch.rand_like(count_cols.float())

            # 2. Add noise only to cells that were at the minimum
            noisy_counts = count_cols.float() + random_noise

            # 3. Mask out non-minimum cells by setting them to a very high value
            noisy_counts[~is_min_mask] = float('inf')

            # 4. Now argmin will pick a random cell from among the original ties
            winner_indices = torch.argmin(noisy_counts, dim=1)

            # Use advanced indexing to set winners to True
            self.burst_winners[remaining_cols_loc, winner_indices] = True

    def _pick_active_burst_cells(self) -> None:
        if self.active_not_pred is not None and self.under_threshold_pred is not None:
            torch.bitwise_and(self.active_not_pred,
                              self.under_threshold_pred,
                              out=self.burst_winners)

    def _pick_random_burst_cells(self, for_cols: torch.Tensor) -> torch.Tensor:
        active_not_pred_winner = torch.randint(
            0,
            self.cells_per_column,
            for_cols.shape,
            dtype=pyu.choose_int_type(self.cells_per_column),
            device=self.device)
        return (for_cols * self.cells_per_column) + active_not_pred_winner

    def _learn(self) -> None:
        active_and_pred_loc = pyu.nonzero_flatten(self.active_and_pred.squeeze())
        if active_and_pred_loc is not None:
            winning_segments = torch.isin(
                self.segment_post,
                active_and_pred_loc
            )
            winning_active_segments = (winning_segments & self.over_threshold_segments)
            if self.debug:
                print("reinforce winning segments", winning_segments.nonzero().numel())
            self._reinforce_segments(winning_active_segments.nonzero().view(-1))

        burst_loc = pyu.nonzero_flatten(self.burst_winners.squeeze())
        if burst_loc is not None:
            winning_segments = torch.isin(
                self.segment_post,
                burst_loc
            )

            winning_active_segments = (winning_segments & self.under_threshold_segments)
            if self.debug:
                print("reinforce burst winners", winning_active_segments.nonzero().numel())
            self._reinforce_segments(winning_active_segments.nonzero().view(-1))

            missing_from_segments = torch.isin(
                burst_loc,
                self.segment_post,
                invert=True
            )
            missing_values = burst_loc[missing_from_segments]
            if self.debug:
                print("creating segments", missing_values.numel())
            for missing_value in missing_values:
                self._create_segment(missing_value)

    def _reinforce_segments(self, segments: torch.Tensor):
        # todo: change to cells?
        if self.previous_active_locations is not None and segments.numel() > 0:
            is_active_mask = torch.isin(self.synapses, self.previous_active_locations)
            is_previous_active = is_active_mask[segments]

            update_deltas = (
                    is_previous_active.float() * self.permanence_inc
                    + (~is_previous_active).float() * self.permanence_dec
            )

            # Apply and clamp
            self.permanences[segments] += update_deltas
            self.permanences[segments].clamp_(min=0.0, max=1.0)  # In-place clamp_

    def _create_segment(self, from_cell: int) -> None:
        if self.previous_active_locations is not None and self.previous_active_locations.numel():
            self.segment_post_all[self.num_segments] = from_cell
            self.column_connection_counts.view(-1)[from_cell] += 1

            sample_size = min(len(self.previous_active_locations), self.segment_size)
            random_indices = torch.randperm(self.previous_active_locations.numel())
            sample_indices = random_indices[:sample_size]

            prev_sample_indices = self.previous_active_locations.view(-1)[sample_indices]
            self.synapses_all[self.num_segments, 0:len(prev_sample_indices)] = prev_sample_indices
            self.permanences_all[self.num_segments, 0:len(prev_sample_indices)] = self.initial_permanence
            self.num_segments += 1
            self.set_segment_shortcuts()

    def _predict_cells(self) -> None:
        self.predicted.zero_()

        active_locations = pyu.nonzero_flatten(self.pred_and_burst_winners.squeeze())
        if active_locations is not None:
            active_synapses_potential = torch.isin(
                self.synapses,
                active_locations
            )

            under_threshold_synapses = (active_synapses_potential & (self.permanences > 0))
            under_threshold_synapses_sum = under_threshold_synapses.sum(dim=1)

            self.under_threshold_segments.zero_()
            under_threshold_segments_indexes = self.pick_best_segment(max_scores=under_threshold_synapses_sum)
            self.under_threshold_segments[under_threshold_segments_indexes] = True
            self.set_segment_shortcuts()
            under_threshold_locations = self.segment_post[under_threshold_segments_indexes]
            pyu.set_from_indices(self.under_threshold_pred, under_threshold_locations)

            over_threshold_synapses = (active_synapses_potential & (
                    self.permanences >= self.permanence_threshold))
            over_threshold_synapses_sum = over_threshold_synapses.sum(dim=1)

            self.over_threshold_segments.zero_()
            over_threshold_segments_indexes = self.pick_best_segment(
                max_scores=over_threshold_synapses_sum,
                threshold=self.segment_threshold)
            self.over_threshold_segments[over_threshold_segments_indexes] = True
            self.set_segment_shortcuts()
            self.predicted_locations = self.segment_post[over_threshold_segments_indexes]
            pyu.set_from_indices(self.predicted, self.predicted_locations)

    def pick_best_segment(self, max_scores: Tensor, threshold: int = 0) -> Tensor:
        self.segment_post_max.scatter_reduce_(
            dim=0,
            index=self.segment_post,
            src=max_scores.to(torch.int32),
            reduce='amax',
            include_self=False
        )
        max_per_element = self.segment_post_max[self.segment_post]
        is_max = (max_scores == max_per_element) & (max_scores >= threshold) & (max_scores > 0)
        winner_indices = torch.nonzero(is_max).view(-1)

        if winner_indices.numel() > 0:
            unique_locs, first_occurrence_mask_idx = torch.unique(
                self.segment_post[winner_indices],
                return_inverse=True
            )

            unique_ids = torch.unique(first_occurrence_mask_idx)
            final_winners = winner_indices[unique_ids]
        else:
            final_winners = torch.empty(0, dtype=torch.long, device=max_scores.device)

        return final_winners
