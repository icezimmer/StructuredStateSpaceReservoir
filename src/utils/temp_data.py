import pickle
import os


def save_temp_data(data, file_name, file_path='../saved_data/'):
    with open(os.path.join(file_path, file_name), 'wb') as file:
        pickle.dump(data, file)


def load_temp_data(file_name, file_path='../saved_data/'):
    with open(os.path.join(file_path, file_name), 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data
