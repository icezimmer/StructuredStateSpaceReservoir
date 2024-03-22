from lra_benchmarks.data.pathfinder import Pathfinder32
from src.torch_dataset.torch_pathfinder import PathfinderDataset
from src.utils.temp_data import save_temp_data
from torch.utils.data import DataLoader, random_split

builder_dataset = Pathfinder32()  # Or Pathfinder64, Pathfinder128, Pathfinder256 depending on your needs
builder_dataset.download_and_prepare()
develop_dataset, test_dataset = builder_dataset.as_dataset(split=['easy[80%:]', 'easy[:20%]'], as_supervised=True)

develop_dataset = PathfinderDataset(develop_dataset, 'cuda:1')
train_size = int(0.8 * len(develop_dataset))
val_size = len(develop_dataset) - train_size
train_dataset, val_dataset = random_split(develop_dataset, [train_size, val_size])
develop_dataloader = DataLoader(develop_dataset, batch_size=128, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = PathfinderDataset(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=128)

save_temp_data(develop_dataloader, 'pathfinder_develop_dataloader')
save_temp_data(train_dataloader, 'pathfinder_train_dataloader')
save_temp_data(val_dataloader, 'pathfinder_val_dataloader')
save_temp_data(test_dataloader, 'pathfinder_test_dataloader')
