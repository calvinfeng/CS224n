import numpy as np
import tensorflow as tf
import os
import zipfile
import collections
import random
import argparse
import sys

from tempfile import gettempdir
from six.moves import urllib
from tensorflow.contrib.tensorboard.plugins import projector


url = 'http://mattmahoney.net/dc/'
vocabulary_size = 50000
data_index = 0


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, 
                                 default=os.path.join(current_path, 'log'),
                                 help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()


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


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)


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

    batch_size = 128
    embedding_size = 128 # Dimension of embedding vector
    skip_window = 1
    num_skips = 2
    num_sampled = 64

    # Construct validation set
    valid_size = 16
    valid_examples = np.random.choice(100, valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
            train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(tf.random_uniform((vocabulary_size, embedding_size), -1, 1))
                embed_lookup = tf.nn.embedding_lookup(embeddings, train_inputs)
            
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal((vocabulary_size, embedding_size), stddev=1 / np.sqrt(embedding_size)))
            
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros((vocabulary_size)))

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed_lookup,
                        num_sampled=num_sampled,
                        num_classes=vocabulary_size))
        
        tf.summary.scalar('loss', loss)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        
        # Compute cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embed_lookup = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embed_lookup, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    num_steps = 100001
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        init.run()
        print 'graph variables initialized'

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {
                train_inputs: batch_inputs, 
                train_labels: batch_labels
            }
            run_metadata = tf.RunMetadata()
            _, summary, loss_val = sess.run([optimizer, merged, loss], feed_dict=feed_dict,
                                                                       run_metadata=run_metadata)
            average_loss += loss_val
            writer.add_summary(summary, step)
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                
                # The average loss is an estimate of the loss over the last 2000 batches.
                print 'average loss at step ', step, ': ', average_loss
                average_loss = 0
        
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        print '%s %s,' % (log_str, close_word)

        final_embeddings = normalized_embeddings.eval()

        with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(current_path, 'tsne.png'))

    except ImportError as ex:
        print 'please install sklearn, matplotlib, and scipy to show embeddings.'
        print ex