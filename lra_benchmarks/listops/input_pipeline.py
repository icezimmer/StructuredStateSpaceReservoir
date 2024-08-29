# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input pipeline for the listops dataset."""

import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

AUTOTUNE = tf.data.experimental.AUTOTUNE


def rename_close_brackets(x):
    source = x['Source']
    source = tf.strings.regex_replace(source, ']', 'X')
    source = tf.strings.regex_replace(source, r'\(', '')
    source = tf.strings.regex_replace(source, r'\)', '')
    return {'Source': source, 'Target': x['Target']}


def preprocess_dataset(file_path, batch_size=256):
    """Preprocess dataset."""
    tf.logging.info(file_path)
    sel_cols = ['Source', 'Target']
    col_defaults = [tf.string, tf.int32]
    ds = tf.data.experimental.make_csv_dataset([file_path],
                                               batch_size,
                                               column_defaults=col_defaults,
                                               select_columns=sel_cols,
                                               field_delim='\t',
                                               header=True,
                                               num_epochs=1)
    ds = ds.unbatch()
    # we rename close brackets to X for this particular task because
    # tokenizer removes non alphanumeric.
    # since there is no trivial way to change this behaviour
    # we opt for an equivalent fix since the vocab in listops is fixed.
    ds = ds.map(rename_close_brackets, num_parallel_calls=AUTOTUNE)
    return ds


def get_datasets(n_devices,
                 task_name,
                 data_dir=None,
                 batch_size=256,
                 max_length=2000):
    """Get algorithmic datasets."""
    if batch_size % n_devices:
        raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                         (batch_size, n_devices))

    develop_path = data_dir + '/{}_develop.tsv'.format(task_name)
    test_path = data_dir + '/{}_test.tsv'.format(task_name)

    develop_dataset = preprocess_dataset(develop_path, batch_size)
    test_dataset = preprocess_dataset(test_path, batch_size)

    tf.logging.info('Finished preprocessing')
    tf.logging.info('Building vocab')
    # Build vocab
    vocab_set = set()
    tokenizer = text.WhitespaceTokenizer()

    lengths = []
    for i, data in enumerate(develop_dataset):
        examples = data['Source']
        examples = tokenizer.tokenize(examples.numpy())
        examples = np.reshape(examples, (-1)).tolist()
        lengths.append(len(examples))
        vocab_set.update(examples)
        if i % 1000 == 0:
	        tf.logging.info('Processed {}'.format(i))
        if i > 1000:
	        break

    # Convert the set to a list
    vocab_list = list(vocab_set)

    # Add a dummy token "<PAD>" to reserve the 0 index
    vocab_list.insert(0, "<PAD>")

    tf.logging.info('Finished processing vocab size={}'.format(len(vocab_list)))

    # Create the TokenTextEncoder with the modified vocabulary list
    encoder = tfds.deprecated.text.TokenTextEncoder(vocab_list)

    def tf_encode(x):
        result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())),
	                        [x],
	                        tf.int32)
        result.set_shape([None])
        return result

    def tokenize(d):
        inputs = tf_encode(d['Source'])

        # Apply padding
        inputs = inputs[:max_length]  # Truncate if necessary
        inputs = tf.pad(inputs, [[0, max_length - tf.shape(inputs)[0]]], constant_values=0)

        return {'inputs': inputs[:max_length], 'targets': d['Target']}

    develop_dataset = develop_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)

    return develop_dataset, test_dataset, encoder
