from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import VanillaRNN
from src.models.s4d.s4d import S4D
from src.models.ssrm.s4dr import S4DR
from src.models.ssrm.s5r import S5R
from src.models.ssrm.s5fr import S5FR
from src.models.ssrm.s4r import S4R
from src.utils.temp_data import save_temp_data, load_temp_data
from src.task.test_classifier import TestClassifier


if __name__ == "__main__":
    NUM_CLASSES = 10
    NUM_FEATURES_INPUT = 1
    KERNEL_SIZE = 28 * 28

    train_dataloader = load_temp_data('smnist_train_dataloader')
    test_dataloader = load_temp_data('smnist_test_dataloader')

    smnist = TestClassifier(block_factory=S4R, device_name='cuda:1', num_classes=NUM_CLASSES, n_layers=2,
                            #d_model=8)
                            d_input=NUM_FEATURES_INPUT, d_state=16384,
                            kernel_size=KERNEL_SIZE, strong_stability=0.8, weak_stability=0.9)

    # for param in smnist.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    smnist.fit_model(num_epochs=50, lr=0.001, train_dataloader=train_dataloader)

    # for param in smnist.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    smnist.evaluate_model(train_dataloader)
    smnist.evaluate_model(test_dataloader)
