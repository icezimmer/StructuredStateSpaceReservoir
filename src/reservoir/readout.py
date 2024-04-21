import torch
from src.utils.check_device import check_data_device


class Offline:
    def __init__(self, model, develop_dataloader):
        self.device = check_data_device(develop_dataloader)
        self.model = model.to(self.device)
        self.develop_dataloader = develop_dataloader
        self.to_vec = next(iter(develop_dataloader))[1].dim() == 1

    # TODO: repair the bug in the code
    def __call__(self):
        with torch.no_grad():
            state_list = []
            label_list = []
            for input_batch, label_batch in self.develop_dataloader:
                state_batch = self.model(input_batch)  # (B, P, L)
                state_batch = state_batch.to('cpu')
                label_batch = label_batch.to('cpu')
                state_list.append(state_batch)
                label_list.append(label_batch)

            state = torch.cat(state_list, dim=0)  # (N, P, L)
            label = torch.cat(label_list, dim=0)  # (N, ?)

            if self.to_vec:
                pooled = state[:, :, -1]  # Taking the last timestep for all instances (N, P)
                target = label.unsqueeze(1).expand(-1, state.shape[-1]).reshape(-1)  # (N * L)
            else:
                pooled = state  # (N, P, L)
                target = label  # Assume (N, L) or needs similar handling

            state = state.reshape(state.shape[0] * state.shape[-1], -1)  # (N * L, P)
            target = target.reshape(-1)  # Flatten target for fitting (N * L,)

            # Convert to numpy arrays for scikit-learn
            pooled = pooled.numpy()
            label = label.numpy()
            state = state.numpy()
            target = target.numpy()

            print('Output shapes for scikit-learn:')
            print('Pooled:', pooled.shape)
            print('Label:', label.shape)
            print('Output:', state.shape)
            print('Target:', target.shape)
            print(state[0, :])
            print(pooled[0, :])

        return pooled, label, state, target
