from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import VanillaRNN
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.utils.temp_data import save_temp_data, load_temp_data
from src.task.test_classifier import TestClassifier


if __name__ == "__main__":
    NUM_CLASSES = 2
    NUM_FEATURES_INPUT = 1

    train_dataloader = load_temp_data('pathfinder_train_dataloader')
    test_dataloader = load_temp_data('pathfinder_test_dataloader')

    pathfinder = TestClassifier(block_factory=S4D, device_name='cuda:1', num_classes=NUM_CLASSES,
                                n_layers=2, d_input=NUM_FEATURES_INPUT, d_state=1024)

    pathfinder.fit_model(num_epochs=15, lr=0.001, train_dataloader=train_dataloader)

    pathfinder.evaluate_model(train_dataloader)
    pathfinder.evaluate_model(test_dataloader)
