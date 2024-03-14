import tensorflow as tf

from lra_benchmarks.data.pathfinder import Pathfinder32
from src.utils.torch_dataset import image_classifier
from src.utils.temp_data import save_temp_data

builder_dataset = Pathfinder32()  # Or Pathfinder64, Pathfinder128, Pathfinder256 depending on your needs
builder_dataset.download_and_prepare()
dataset_train, dataset_test = builder_dataset.as_dataset(split=['easy[80%:]', 'easy[:20%]'], as_supervised=True,
                                                         shuffle_files=True)
batch_size = 128
dataset_train = dataset_train.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
train_dataloader = image_classifier(dataset_train)
test_dataloader = image_classifier(dataset_test)
save_temp_data(train_dataloader, 'train_dataloader')
save_temp_data(test_dataloader, 'test_dataloader')
