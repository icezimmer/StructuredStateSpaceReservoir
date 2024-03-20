from lra_benchmarks.data.pathfinder import Pathfinder32
from src.torch_dataset.torch_pathfinder import PathfinderDataset
from src.utils.temp_data import save_temp_data
from torch.utils.data import DataLoader

builder_dataset = Pathfinder32()  # Or Pathfinder64, Pathfinder128, Pathfinder256 depending on your needs
builder_dataset.download_and_prepare()
dataset_train, dataset_test = builder_dataset.as_dataset(split=['easy[80%:]', 'easy[:20%]'], as_supervised=True)

dataset_train = PathfinderDataset(dataset_train, 'cuda:1')
train_dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_test = PathfinderDataset(dataset_test, 'cuda:1')
test_dataloader = DataLoader(dataset_test, batch_size=128, shuffle=False)

save_temp_data(train_dataloader, 'pathfinder_train_dataloader')
save_temp_data(test_dataloader, 'pathfinder_test_dataloader')
