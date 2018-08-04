import numpy as np
import tensorflow as tf


def read_and_decode(filename,batch_size,max_length): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'input_sequence' : tf.FixedLenFeature([max_length],tf.int64),
                                           'predict_label' : tf.FixedLenFeature([],tf.int64)
                                       })#return image and label
    input_sequence = features['input_sequence']
    predict_label = tf.cast(features['predict_label'], tf.int32)
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    input_batch, predict_batch , label_batch = tf.train.shuffle_batch([input_sequence , predict_label, label],
                                                    batch_size= batch_size,
                                                    num_threads=64,
                                                    capacity=2000,
                                                    min_after_dequeue=1500,
                                                    )
    return tf.reshape(input_batch,[batch_size,max_length]), tf.reshape(predict_batch,[batch_size]), tf.reshape(label_batch,[batch_size])

class DataHelper(object):
    def __init__(self):
        pass
    @staticmethod
    def read_and_decode(file_list, batch_size):
        filename_queue = tf.train.string_input_producer(file_list)  # create a queue
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # return file_name and file
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'user_sex': tf.FixedLenFeature([], tf.int64),
                'user_age': tf.FixedLenFeature([], tf.int64),
                'user_region': tf.FixedLenFeature([], tf.int64),
                'user_vid_list': tf.VarLenFeature(tf.int64),
                'user_vid_list_value': tf.VarLenFeature(tf.float32),
                'user_tag_list':tf.VarLenFeature(tf.int64),
                'user_tag_list_value': tf.VarLenFeature(tf.float32),
                'user_subcategory_list': tf.VarLenFeature(tf.int64),
                'user_subcategory_list_value': tf.VarLenFeature(tf.float32),
                'doc_vid': tf.FixedLenFeature([], tf.int64),
                'doc_tag_list': tf.VarLenFeature(tf.int64),
                'doc_tag_list_value': tf.VarLenFeature(tf.float32),
                'doc_subcategory': tf.FixedLenFeature(tf.int64)
            })  # return image and label
        label = features['label']
        user_sex = features['user_sex']
        user_age = features['user_age']
        user_region = features['user_region']
        doc_vid = features['doc_vid']
        doc_subcategory = features['doc_subcategory']
        label, user_sex, user_region, doc_vid, doc_subcategory, features = \
            tf.train.shuffle_batch([label, user_sex, user_age, user_region, doc_vid, doc_subcategory, features],
                                   batch_size=batch_size,
                                   num_threads=8,
                                   capacity=200000,
                                   min_after_dequeue=1500+4*batch_size
                                   )
        label = label.reshape([batch_size])
        user_sex = user_sex.reshpae([batch_size])
        user_age = user_age.reshape([batch_size])
        user_region = user_region.reshape([batch_size])
        doc_vid = doc_vid.reshape([batch_size])
        doc_subcategory = doc_subcategory.reshape([batch_size])
        return label, user_sex, user_sex, user_region, doc_vid, doc_subcategory, features

        
if __name__ == "__main__":
    DataHelper.read_and_decode()
    DataHelper.read_and_decode()