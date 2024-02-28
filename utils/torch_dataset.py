import torch
import tensorflow as tf


def image_classifier(dataset):
    global mode, batch_size, num_input_features, length
    torch_input_list = []
    torch_label_list = []

    for input_batch, _ in dataset.take(1):
        if len(input_batch.shape) == 4:
            mode = 'batch'
            batch_size = input_batch.shape[0]
            num_input_features = input_batch.shape[3]
            length = input_batch.shape[1] * input_batch.shape[2]
        # No batch mode (test)
        elif len(input_batch.shape) == 3:
            mode = 'no_batch'
            batch_size = 1
            num_input_features = input_batch.shape[2]
            length = input_batch.shape[0] * input_batch.shape[1]

    for input_batch, label_batch in dataset:
        # input_batch has shape (batch_size, num_rows, num_columns, num_input_features)

        # Create torch tensor input of size (B, H, L) = (batch_size,
        #                                               num_input_features = num_channels,
        #                                               length = num_rows * num_columns)
        torch_input = torch.zeros(batch_size,
                                  num_input_features,
                                  length)

        # Create torch tensor label of size (B, L) = (batch_size, length = 1)
        torch_label = torch.zeros(batch_size, 1)

        for i in range(min(batch_size, input_batch.shape[0])):
            # tf_input has shape (num_columns, num_rows, num_features_input)
            if mode == 'batch':
                input_ = input_batch[i, :, :, :]
            elif mode == 'no_batch':
                input_ = input_batch
            input_ = tf.reshape(input_, (num_input_features, length))
            input_ = tf.expand_dims(input_, axis=0)  # Add batch dimension
            input_ = torch.from_numpy(input_.numpy())
            torch_input[i, :, :] = input_

            if mode == 'batch':
                label = label_batch[i]
            if mode == 'no_batch':
                label = label_batch
            label = tf.expand_dims(label, axis=0)  # Add batch dimension
            label = torch.from_numpy(label.numpy())
            torch_label[i, :] = label

        torch_input_list.append(torch_input)
        torch_label_list.append(torch_label)

    return torch_input_list, torch_label_list
