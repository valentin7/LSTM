from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm

import tensorflow as tf
import numpy as np
import re
import operator
import math

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

def main():
    words_dict, all_words, words_vector = getWords()

    words_len = len(words_vector)
    trainWords = words_vector[0:int(words_len*(9/10))]
    testWords = words_vector[int(words_len*(9/10)):]
    #words_vector = getWordsVector(all_words, words_dict)
    runModel(words_dict, trainWords, testWords)

def getWords():
    words_dict = {}
    counter = 0
    all_total = 0
    all_words = []
    word_frequencies = {}

    with open("sherlock.txt", "r") as ins:
        for line in ins:
            tokens = basic_tokenizer(line)
            #print("tokens in line: ", tokens)
            #tokens.append("STOP")
            for token in tokens:
                word = token
                all_words.append(word)
                all_total += 1

                if not words_dict.has_key(word):
                    words_dict[word] = counter
                    word_frequencies[word] = 1
                    counter += 1
                else:
                    word_frequencies[word] = word_frequencies[word] + 1

    print("all total is ", all_total)
    print("counter is ", counter)

    sorted_frequencies = sorted(word_frequencies.items(), key=operator.itemgetter(1), reverse=True)
    last_8000_word_frequency = sorted_frequencies[7999]
    num_extra_words_with_same_freq = 0
    min_freq = last_8000_word_frequency[1]
    num_words_allowed_with_lowest_freq = 0

    print("min_freq is ", min_freq)
    #print("SORTED FREQUENCIES", sorted_frequencies)

    i = 7999
    while (sorted_frequencies[i][1] == min_freq):
        i -= 1
        num_words_allowed_with_lowest_freq += 1

    print("last 8000 word freq:", last_8000_word_frequency)
    print("words allowed with same freq:", num_words_allowed_with_lowest_freq)

    words_no_unk = 0
    words_vector = []


    for index, word in enumerate(all_words):
        #replace unfrequent words with UNK
        if word_frequencies[word] <= min_freq:

            #case where some of them are still allowed with the lowest frequency
            if word_frequencies[word] == min_freq:
                if num_words_allowed_with_lowest_freq > 0:
                    num_words_allowed_with_lowest_freq -= 1
                    words_no_unk += 1
                else:
                    all_words[index] = "UNK"
                    del words_dict[word]
                    words_dict[word] = -1

            else:
                all_words[index] = "UNK"
                del words_dict[word]
                words_dict[word] = -1
        else:
            words_no_unk += 1

        #convert word to number (id)
        if words_dict.has_key(word):
            words_vector.append(words_dict[word])
        else:
            print("didn't find word in dict")

    print("total_words =", len(all_words))
    print("words no unk ", words_no_unk)

    return words_dict, all_words, words_vector

def getWordsVector(all_words, words_dict):
    words_vector = []

    for word in all_words:
        if words_dict.has_key(word):
            words_vector.append(words_dict[word])
        else:
            print("didn't find word in dict")
    return words_vector

def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]

def runModel(words_dict, trainWords, testWords):
    sess = tf.InteractiveSession()
    # Create the model
    vocab_len = len(words_dict)

    print("vocab_len is ", vocab_len)
    #x = tf.placeholder(tf.int32, [vocab_len])

    # Define some parameters
    embed_size = 50
    hSize = 100
    batch_size = 20
    num_steps = 20
    lstm_cell_size = 256
    keep_prob = 0.5

    input_words = tf.placeholder(tf.int32, [batch_size, num_steps])
    output_words = tf.placeholder(tf.int32, [batch_size, num_steps])
    keep_prob = tf.placeholder(tf.float32)

    embedding_matrix = weight_variable([vocab_len, embed_size])#tf.Variable(tf.random_uniform([vocab_len, embed_size], -0.1, 0.1))

    basicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_cell_size, forget_bias=1.0, input_size=None, state_is_tuple=True)
    initial_state = basicLSTMCell.zero_state(batch_size, tf.float32)

    print("full lstm initial state is ", initial_state)
    print("second item of initial state is ", initial_state[1])

    W_conv1 = weight_variable([lstm_cell_size, vocab_len])
    b_conv1 = bias_variable([vocab_len])

    embed_lookup = tf.nn.embedding_lookup(embedding_matrix, input_words)
    embeds_drop = tf.nn.dropout(embed_lookup, keep_prob)

    outputs, f_state = tf.nn.dynamic_rnn(basicLSTMCell, embeds_drop, initial_state=initial_state)
    outputs_2d = tf.reshape(outputs, [batch_size*num_steps, lstm_cell_size])
    logits = tf.matmul(outputs_2d, W_conv1) + b_conv1

    #print("logits length is ", len(logits))
    #print("output_words length is ", len(output_words))

    ## receives the logits, targets, and weights.
    weights = tf.ones([batch_size*num_steps], tf.float32)
    output_words_1D =  tf.reshape(output_words, [batch_size*num_steps])

    #loss calculation
    all_losses = tf.nn.seq2seq.sequence_loss_by_example([logits], [output_words], [weights])

    avg_loss = tf.reduce_sum(all_losses)/batch_size
    #training operation
    train_step = tf.train.AdamOptimizer(1e-4).minimize(avg_loss)

    sess.run(tf.initialize_all_variables())
    #state_eval = sess.run(f_state)

    batches = 0
    total_loss = 0
    all_length = len(trainWords)

    state = initial_state
    state_eval = sess.run(state)

    print("starting train, ALL train words length is ", all_length)

    prev_perplexity = 0
    times_not_changed = 0

    #Train
    for epoch in tqdm(range(10)):
        for i in tqdm(range(0, all_length - batch_size-2, batch_size*num_steps), leave=False):
            batch = trainWords[i:i+(batch_size*num_steps)]
            batches += 1
            next_words = trainWords[i+1:i+(batch_size*num_steps)+1]

            # if we're at the end of the array and we don't have enough next_words for a correct shape
            if len(next_words) != batch_size*num_steps:
                break

            batch_with_num_steps = np.reshape(batch, (batch_size, num_steps))#tf.reshape(batch, [batch_size, num_steps])
            next_words_with_num_steps = np.reshape(next_words, (batch_size, num_steps))#tf.reshape(next_words, [batch_size, num_steps])

            feed_dict_ = {input_words: batch_with_num_steps, output_words: next_words_with_num_steps, keep_prob: 0.5, initial_state[0]: state_eval[0], initial_state[1]: state_eval[1]}
            _, state, loss_  = sess.run([train_step, f_state, avg_loss], feed_dict=feed_dict_)
            state_eval = state

            total_loss += loss_
            perplexity = np.exp(total_loss/(num_steps * batches))

            convergence_diff = 1
            if (abs(prev_perplexity - perplexity) < convergence_diff):
                times_not_changed += 1
                print("NOT CHANGED MORE THAN ", convergence_diff)
                print("prev perplexity ", prev_perplexity)
                print("current perplexity ", perplexity)

                if (times_not_changed > 30):
                    print("CONVERGED at batch", i)
                    print("Epoch", epoch)
                    print("train perplexity is ", perplexity)
                    break
            else:
                times_not_changed = 0

            prev_perplexity = perplexity
            #initial_state = state
            if i%100 == 0:
                print("batch is ", i)
                print("train perplexity is ", perplexity)


    print("final train perplexity is ", perplexity)
    print("total batches is ", batches)

    test_batches = 0
    test_total_loss = 0
    test_all_length = len(testWords)

    print("test total length is ", test_all_length)
    
    for i in tqdm(range(0, test_all_length - batch_size-2, batch_size*num_steps), leave=False):
        batch = testWords[i:i+(batch_size*num_steps)]
        test_batches += 1
        next_words = testWords[i+1:i+(batch_size*num_steps)+1]

        print("in TEST batch ", i)
        # if we're at the end of the array and we don't have enough next_words for a correct shape
        if len(next_words) != batch_size*num_steps or len(batch) != batch_size*num_steps:
            break

        batch_with_num_steps = np.reshape(batch, (batch_size, num_steps))#tf.reshape(batch, [batch_size, num_steps])
        next_words_with_num_steps = np.reshape(next_words, (batch_size, num_steps))#tf.reshape(next_words, [batch_size, num_steps])

        print("RESHAPED, on feedict")
        feed_dict_ = {input_words: batch_with_num_steps, output_words: next_words_with_num_steps, keep_prob: 0.5, initial_state[0]: state_eval[0], initial_state[1]: state_eval[1]}
        _, state, loss_  = sess.run([f_state, avg_loss], feed_dict=feed_dict_)
        state_eval = state

        test_total_loss += loss_
        perplexity = np.exp(test_total_loss/(num_steps * batches))
        if i%100 == 0:
            print("batch is ", i)
            print("test perplexity is ", perplexity)


    print("final test perplexity is ", perplexity)
    print("total test batches is ", test_batches)
    # perplexity_test = tf.exp(cross_entropy_loss)
    #
    # print("test perplexity is: ", perplexity_test.eval(feed_dict={input_words: test_words_vector[:-1], output_words: test_words_vector[1:]}))


def getWithNumSteps(batch, numSteps):
    withNumSteps = []
    for i in range(0,numSteps):
        withNumSteps.append(batch)

    return withNumSteps

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    main()
