from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import VanillaRNN
from src.models.s4d.s4d import S4D
from src.models.ssrm.s5r import S5R
from src.models.ssrm.s4r import S4R
from src.utils.temp_data import save_temp_data, load_temp_data
from src.task.test_classifier import TestClassifier


if __name__ == "__main__":
    develop_dataloader = load_temp_data('pathfinder_develop_dataloader')
    train_dataloader = load_temp_data('pathfinder_train_dataloader')
    val_dataloader = load_temp_data('pathfinder_val_dataloader')
    test_dataloader = load_temp_data('pathfinder_test_dataloader')

    NUM_CLASSES = 2
    NUM_FEATURES_INPUT = 1
    KERNEL_SIZE = 1024
    DEVICE = next(iter(train_dataloader))[0].device

    pathfinder = TestClassifier(block_factory=S4Block, device=DEVICE, num_classes=NUM_CLASSES, n_layers=3,
                                d_model=256)
                                # d_input=NUM_FEATURES_INPUT, d_state=512*4,
                                # kernel_size=KERNEL_SIZE, strong_stability=0.99, weak_stability=1.0)

    pathfinder.fit_model(lr=0.004, develop_dataloader=develop_dataloader, patience=10,
                         train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                         name='pathfinder_model')

    pathfinder.evaluate_model(develop_dataloader)
    pathfinder.evaluate_model(test_dataloader)
