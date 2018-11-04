import numpy as np
import tensorflow as tf
import os
import zipfile
import collections
import random

from tempfile import gettempdir
from six.moves import urllib


url = 'http://mattmahoney.net/dc/'
vocabulary_size = 50000
data_index = 0


def download(filename, expected_bytes):
    """Download a zip folder to temp folder on Linux"""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url+filename, local_filename)

    statinfo = os.stat(local_filename)
    if statinfo.st_size != expected_bytes:
        raise Exception('wrong size ' + statinfo.st_size + ' failed to verify ' + filename)
    
    return local_filename


def read_data(filename):
    """Extract the file file enclosed in a zip file as a list of strings"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    
    return data


def build_dataset(words, vocab_size):
    """Builds a dataset, which is a list of integers. Each integer maps to a word."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0: # Equivalent to dictionary['UNK']
            unk_count += 1
        data.append(index)
    
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(raw_data, batch_size, num_skips, skip_window):
    """Generate a training batch for the skip-gram model"""
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [skip_window target skip_window]
    buffer = collections.deque(maxlen=span) 

    if data_index + span > len(raw_data):
        data_index = 0

    buffer.extend(raw_data[data_index:data_index+span])
    data_index += span
    for i in range(batch_size//num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j] = buffer[context_word]

        if data_index == len(data):
            buffer.extend(raw_data[0:span])
            data_index = span
        else:
            buffer.append(raw_data[data_index])
            data_index += 1

    data_index = (data_index + len(raw_data) - span) % len(raw_data)
    return batch, labels


if __name__ == '__main__':
    filename = download('text8.zip', 31344016)
    print 'found and verified %s' % filename
    word_list = read_data(filename)
    print 'word count: %d' % len(word_list)
    data, count, dictionary, reverse_dictionary = build_dataset(word_list, vocabulary_size)
    del word_list
    print 'most common words (+UNK)', count[:5]
    print 'sample data', data[:10], [reverse_dictionary[i] for i in data[:10]]
    batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]]