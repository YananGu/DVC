import torch
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.buffer_utils import ClassBalancedRandomSampling, random_retrieve
from utils.buffer.aser_utils import compute_knn_sv, add_minority_class_input
from utils.setup_elements import n_classes
from utils.utils import nonzero_indices, maybe_cuda


class ASER_update(object):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = params.k
        self.mem_size = params.mem_size
        self.num_tasks = params.num_tasks
        self.out_dim = n_classes[params.data]
        self.n_smp_cls = int(params.n_smp_cls)
        self.n_total_smp = int(params.n_smp_cls * self.out_dim)
        self.reservoir_update = Reservoir_update(params)
        ClassBalancedRandomSampling.class_index_cache = None

    def update(self, buffer, x, y, **kwargs):
        model = buffer.model

        place_left = self.mem_size - buffer.current_index

        # If buffer is not filled, use available space to store whole or part of batch
        if place_left:
            x_fit = x[:place_left]
            y_fit = y[:place_left]

            ind = torch.arange(start=buffer.current_index, end=buffer.current_index + x_fit.size(0), device=self.device)
            ClassBalancedRandomSampling.update_cache(buffer.buffer_label, self.out_dim,
                                                     new_y=y_fit, ind=ind, device=self.device)
            self.reservoir_update.update(buffer, x_fit, y_fit)

        # If buffer is filled, update buffer by sv
        if buffer.current_index == self.mem_size:
            self.reservoir_update.update(buffer, x_fit, y_fit)


        ClassBalancedRandomSampling.update_cache(buffer.buffer_label, self.out_dim,
                                                 new_y=y_upt, ind=ind_buffer, device=self.device)

