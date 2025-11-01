import time
import torch
import torch.nn as nn

from torchsde import sdeint


class SDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.sde_type = "ito"
        self.noise_type = "scalar"
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)
        self.drift = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Softplus(),  # 保证扩散项非负
            nn.Linear(64, hidden_dim)
        )

        for layer in self.drift:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

        for layer in self.diffusion:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def f(self, t, h):
        return self.drift(h)

    def g(self, t, h):
        return self.diffusion(h).unsqueeze(-1)  # [batch_size, hidden_dim, noise_dim]

    def forward(self, inputs, times, non_zero):
        # start_time = time.time()
        seq_len = inputs.size()[0]
        batch_size = inputs.size()[1]
        hidden_dim = inputs.size()[2]
        # y = [inputs[0]]  # len = 5 batch = 4 dim = 8
        h = [torch.zeros(batch_size, hidden_dim, requires_grad=True, device='cuda')]  # batch = 4 dim = 8
        h1 = [torch.zeros(batch_size, hidden_dim, requires_grad=True, device='cuda')]
        h2 = [torch.zeros(batch_size, hidden_dim, requires_grad=True, device='cuda')]
        time_diff_list = []
        for i in range(0, seq_len - 1):
            mask = (i < non_zero - 1)  # [batch_size,]
            if not mask.any():
                break
            active_inputs = inputs[i, mask]  # [batch_size, embedding]
            active_h = h2[-1].clone()
            new_active_h = self.rnn_cell(active_inputs, h2[-1][mask])
            active_h[mask] = new_active_h
            h.append(active_h)
            if mask.any():
                batch_indices = torch.where(mask)[0]
                valid_indices = [j for j in batch_indices if times[j, i + 1] != 0]
                if valid_indices:
                    valid_indices = torch.stack(valid_indices)
                    time_span_start = sum(times[j, i] for j in valid_indices) / len(valid_indices) * 0.02
                    time_span_end = sum(times[j, i + 1] for j in valid_indices) / len(valid_indices) * 0.02
                    time_span = torch.tensor([time_span_start, time_span_end], device='cuda').float()
                    if time_span[1] == time_span[0]:
                        time_span[1] += 1
                    time_diff = float(time_span[1] - time_span[0])
                    time_diff_list.append(time_diff)
                    batch_h_j = h[-1][valid_indices]  # [active, hidden_dim]

                    # b_y_j = sdeint(self, batch_h_j, time_span, method='euler', adaptive=True, rtol=1e-3, atol=1e-4)

                    batch_h1_j = sdeint(self, batch_h_j, time_span, method='euler', dt=time_diff/10)
                    h1.append(torch.zeros_like(h1[0]))

                    # h1[-1][valid_indices] = batch_h1_j
                    for idx, j in enumerate(valid_indices):
                        h1[-1][j] = batch_h1_j[-1, idx].squeeze(0)

                    if i != 0:
                        u = 1 / torch.exp(torch.tensor(time_diff_list[-2]))
                        batch_h2_j = u * batch_h_j + (1 - u) * h1[-2][valid_indices]
                        h2.append(torch.zeros_like(h2[0]))
                        for idx, j in enumerate(valid_indices):
                            h2[i + 1][j] = batch_h2_j[-1, idx].squeeze(0)
                    else:
                        h2.append(torch.zeros_like(h2[0]))
                        h2[i + 1] = h[-1]
                    '''if len(y) == i + 1:
                        y.append(torch.zeros_like(inputs[0]))'''

        sde_output = []
        for k in range(batch_size):
            last_valid_idx = non_zero[k] - 1
            if last_valid_idx < 0:
                last_valid_idx = 0
            sde_output.append(h2[last_valid_idx][k])
        stacked_output = torch.stack(sde_output)
        # end_time = time.time()
        # print(end_time - start_time)
        return stacked_output  # [batch_size, hidden_dim]
