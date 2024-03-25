from src.models.s4.s4 import S4Block
from src.models.rnn.vanilla_rnn import VanillaRNN
from src.models.s4d.s4d import S4D
from src.models.ssrm.s4dr import S4DR
from src.models.ssrm.s5r import S5R
from src.models.ssrm.s5fr import S5FR
from src.models.ssrm.s4r import S4R
from src.models.ssrm.s4v import S4V
from src.utils.temp_data import save_temp_data, load_temp_data
from src.task.test_classifier import TestClassifier


if __name__ == "__main__":
    develop_dataloader = load_temp_data('smnist_develop_dataloader')
    train_dataloader = load_temp_data('smnist_train_dataloader')
    val_dataloader = load_temp_data('smnist_val_dataloader')
    test_dataloader = load_temp_data('smnist_test_dataloader')

    NUM_CLASSES = 10
    NUM_FEATURES_INPUT = 1
    KERNEL_SIZE = 28 * 28
    DEVICE = next(iter(develop_dataloader))[0].device

    smnist = TestClassifier(block_factory=S4R, device=DEVICE, num_classes=NUM_CLASSES, n_layers=1,
                            # d_model=128)
                            d_input=NUM_FEATURES_INPUT, d_state=64*64*4,
                            kernel_size=KERNEL_SIZE, strong_stability=0.5, weak_stability=0.5)

    # for param in smnist.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    smnist.fit_model(lr=0.001, develop_dataloader=develop_dataloader, num_epochs=10,
                     train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                     name='smnist_model')

    # for param in smnist.model.parameters():
    #     print(param.data.shape)
    #     print(param)

    smnist.evaluate_model(develop_dataloader)
    smnist.evaluate_model(test_dataloader)
