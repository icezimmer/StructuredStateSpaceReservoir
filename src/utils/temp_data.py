import pickle


def save_temp_data(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_temp_data(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data
