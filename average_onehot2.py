#author: jackqiu

import tensorflow as tf
import numpy as np
import re
import time
import math
from tensorflow.python.ops import variable_scope as vs
import tensor_tools


def full_connected_layer_auto_reuse(input, W_size, b_size, w_name, b_name):
    with tf.variable_scope("full_connected_layer", reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name=w_name, shape=W_size, initializer=tf.contrib.layers.xavier_initializer())
        # W = tf.Variable(tf.random_uniform(W_size, -1, 1))
        b = tf.Variable(tf.constant(0.1, shape=b_size, name=b_name))
        output = tf.nn.xw_plus_b(input, W, b)
    return W, b, output


def full_connected_layer(step, input, W_size, b_size, w_name, b_name):
    print(step)
    reuse_flag = False
    if tf.equal(step, tf.constant(0)) == False:
        reuse_flag = True
    print(reuse_flag)
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_flag):
        W = tf.get_variable(name=w_name, shape=W_size, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=b_size, name=b_name))
        output = tf.nn.xw_plus_b(input, W, b)
    return W, b, output

def random_softmax_loss(nb_negative, inputs, targets, weights, biases=None):
    nb_classes, real_batch_size = tf.shape(weights)[0], tf.shape(targets)[0]
    negative_sample = tf.random_uniform([real_batch_size, nb_negative], 0, nb_classes, dtype=tf.int32)
    random_sample = tf.concat([targets, negative_sample], axis=1)
    sampled_weights = tf.nn.embedding_lookup(weights, random_sample)
    if biases:
        sampled_biases = tf.nn.embedding_lookup(biases, random_sample)
        sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:, 0, :] + sampled_biases
    else:
        sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:, 0, :]
    sampled_labels = tf.zeros([real_batch_size], dtype=tf.int32)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sampled_logits, labels=sampled_labels))


def random_softmax_loss_norm(nb_negative, inputs, targets, weights, biases=None, negative_weight=None):
    nb_classes, real_batch_size = tf.shape(weights)[0], tf.shape(targets)[0]
    # square1 = tf.sqrt(tf.reduce_sum(tf.square()))
    line_square = tensor_tools.get_line_square(inputs, axis=2)
    negative_sample = tf.random_uniform([real_batch_size, nb_negative], 0, nb_classes, dtype=tf.int32)
    random_sample = tf.concat([targets, negative_sample], axis=1)
    sampled_weights = tf.nn.embedding_lookup(weights, random_sample)
    if biases:
        sampled_biases = tf.nn.embedding_lookup(biases, random_sample)
        sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:, 0, :] + sampled_biases
    else:
        sampled_logits = tf.matmul(inputs, sampled_weights, transpose_b=True)[:, 0, :]
    # sampled_logits = tensor_tools.line_divide(sampled_logits, line_square)
    # if negative_weight is not None:
    #    sampled_logits = sampled_logits * []
    sampled_labels = tf.zeros([real_batch_size], dtype=tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sampled_logits,
                                                                   labels=sampled_labels) / line_square
    return tf.reduce_mean(cross_entropy)


def cosine(input_a, input_b):
    square1 = tf.sqrt(tf.reduce_sum(tf.square(input_a), axis=1))
    square2 = tf.sqrt(tf.reduce_sum(tf.square(input_b), axis=1))
    inner_product = tf.reduce_sum(tf.multiply(input_a, input_b), axis=1)
    inner_product = tf.divide(inner_product, square1)
    inner_product = tf.divide(inner_product, square2)
    return inner_product


class AOModel(object):
    def build_graph(self, step, input, output_size=128):
        shape_size = input.get_shape().as_list()
        W_size = [shape_size[1], output_size]
        b_size = [output_size]
        W, b, output = full_connected_layer_auto_reuse(input, W_size=W_size, b_size=b_size, w_name="first_layer_w_jack",
                                                       b_name="first_layer_b_jack")
        self.WWW = W
        self.bbb = b
        output = tf.nn.tanh(output)
        self.l2_loss += tf.nn.l2_loss(W)
        self.l2_loss += tf.nn.l2_loss(b)
        return output

    def most_similar(self, embeddings, topn=10):
        word_vec = embeddings
        word_sim = np.dot(self.W, word_vec)
        word_sim_argsort = word_sim.argsort()[::-1]
        return [(self.item_dict[i], word_sim[i]) for i in word_sim_argsort[0:topn + 1]]

    def read_embedding_file(self, fileName):
        W = []
        with open(fileName, 'r') as reader:
            for line in reader:
                line = line.strip()
                tokens = line.split("\t")
                W.append([float(x) for x in tokens[1].split(":")])
        return tf.Variable(initial_value=W, dtype=tf.float32, trainable=False)

    def __init__(self, age_vocab_size, sex_vocab_size, sequence_length, num_classes, initial_embedding_file, vocab_size,
                 embedding_size,
                 sex_dict,
                 age_dict,
                 item_dict,
                 filter_size=None,
                 num_filters=None,
                 l2_reg_lambda=0.0,
                 trainable=False):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initial_embedding_file = initial_embedding_file
        self.sex_dict = sex_dict
        self.age_dict = age_dict
        self.item_dict = item_dict
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.age_vocab_size = age_vocab_size
        self.sex_vocab_size = sex_vocab_size
        self.l2_reg_lambda = l2_reg_lambda
        self.input_x_sequence = tf.placeholder(tf.int32, [None, sequence_length], name='input_x_seq')
        self.input_c_sequence = tf.placeholder(tf.int32, [None], name="input_c_seq")
        self.input_item_lengths = tf.placeholder(tf.int32, [None], name="input_item_lengths")
        self.step = tf.placeholder(tf.int32, name="step")
        item_sequence = self.input_x_sequence[:, 2:]
        age_profile_info = self.input_x_sequence[:, 0:1]
        sex_profile_info = self.input_x_sequence[:, 1:2]
        self.age_profile_info = tf.one_hot(age_profile_info, depth=age_vocab_size, on_value=1, axis=2)
        self.sex_profile_info = tf.one_hot(sex_profile_info, depth=sex_vocab_size, on_value=1, axis=2)
        self.age_profile_info = tf.cast(self.age_profile_info, tf.float32)
        self.sex_profile_info = tf.cast(self.sex_profile_info, tf.float32)
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size + 1, self.embedding_size])
        self.l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size + 1, self.embedding_size]),
                                 trainable=trainable, name="W")
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            print("assign pretrained embedding fininshed")
            self.embed_x_sequence = tf.nn.embedding_lookup(self.W, item_sequence)
            self.embed_x_sequence = tf.reshape(self.embed_x_sequence, shape=[-1, sequence_length - 2, embedding_size])
            self.tmp_embed_x_sequence = self.embed_x_sequence

        with tf.name_scope("item_embedding_transform"):
            cur_input_item_lengths = tf.divide(1.0, (tf.cast(self.input_item_lengths, tf.float32) + 0.01))
            dig_item_lengths = tf.cast(tf.matrix_diag(cur_input_item_lengths, name="dig_item_lengths"), tf.float32)
            self.dig_il = dig_item_lengths
            self.embed_x_sequence = tf.reduce_sum(self.embed_x_sequence, axis=1)
            self.embed_x_sequence = tf.matmul(dig_item_lengths, self.embed_x_sequence)

        with tf.name_scope("concat_layer"):
            self.sex_profile_info = tf.reshape(self.sex_profile_info, shape=[-1, self.sex_vocab_size])
            self.age_profile_info = tf.reshape(self.age_profile_info, shape=[-1, self.age_vocab_size])
            profile_concat_conv = tf.concat([self.sex_profile_info, self.age_profile_info, self.embed_x_sequence],
                                            axis=-1, name="concat_profile_item")
            self.pcc = profile_concat_conv
            full_connect_output = self.build_graph(self.step, profile_concat_conv, output_size=self.embedding_size)
            self.fco = full_connect_output

        with tf.name_scope("loss"):
            full_connect_output = tf.expand_dims(full_connect_output, 1)
            self.loss = random_softmax_loss(10, full_connect_output, tf.reshape(self.input_c_sequence, [-1, 1]), self.W)


