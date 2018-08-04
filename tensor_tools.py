#author: jackqiu

import tensorflow as tf
from tensorflow.contrib import layers

layers.fully_connected


def line_multiply(input, dig_nums):
    dig = tf.cast(tf.matrix_diag(dig_nums), tf.float32)
    ret = tf.matmul(dig, input)
    return ret


def column_multiply(input, dig_nums):
    dig = tf.cast(tf.matrix_diag(dig_nums), tf.float32)
    ret = tf.matmul(input, dig)
    return ret


def line_divide(input, dig_nums):
    dig_nums = 1.0 / dig_nums
    return line_multiply(input, dig_nums)


def get_line_square(input, axis=1):
    return tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(input), axis=axis)), [-1])


import numpy as np

if __name__ == '__main__':
    rand_matrix = np.random.rand(2, 3)

    print(rand_matrix)
    data_tf = tf.convert_to_tensor(rand_matrix, np.float32)
    sess = tf.Session()
    print(sess.run(line_multiply(data_tf, tf.constant([1, 2.0]))))
    print(sess.run(line_divide(data_tf, tf.constant([1, 2.0]))))
    print(sess.run(get_line_square(data_tf, axis=1)))
