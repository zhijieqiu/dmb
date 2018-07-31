import os
import tensorflow as tf
import numpy as np



def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
def _int64_feature_array(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(byte_list = tf.train.BytesList(value=[value]))
def write_tf(input,output,max_length):
    writer = tf.python_io.TFRecordWriter(output)
    with open(input,'r') as reader:
        index = 0
        for line in reader:
            line = line.strip()
            query_title = line.split("&&")
            label = int(query_title[0])
            input_sequence = [int(x) for x in query_title[1].split("\t")]
            predict_label = int(query_title[2])
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": _int64_feature(label),
                "input_sequence": _int64_feature_array(input_sequence),
                "predict_label":_int64_feature(predict_label)
            }))
            writer.write(example.SerializeToString())
            index+=1
            if index%10000 == 0:
                print(index)
        writer.close()

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

    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img = tf.reshape(img, [208, 208, 3])  #reshape image to 208*208*3

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
def handle_sparse(sparse_matrix,batch_size):
    ret = [[] for i in range(batch_size)]
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    for i in range(np.shape(indices)[0]):
        index = indices[i]
        ret[index[0]].append(values[i])
    return ret
def batch_iter(tf_record_file_name,batch_size,max_length,negative_weight=0.2,max_iteration=10000000000):
    #write_tf(train_file, r'tf.records',max_length)
    input_batch, predict_batch, label_batch = read_and_decode(tf_record_file_name,batch_size, max_length)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i< max_iteration:
                inputs, predicts, label = sess.run([input_batch, predict_batch, label_batch])
                flag = False
                for i in range(batch_size):
                    for j in range(max_length):
                        if inputs[i,j]<0 or inputs[i,j]>=1860668:
                            flag = True
                            break

                    if flag:
                        break
                    if predicts[i]<0 or predicts[i]>=1860668:
                        flag = True
                        break
                    if label[i] not in [0,1]:
                        flag = True
                        break
                if flag:
                    continue
                lable_class = []
                for i in label:
                    if i==0:
                        lable_class.append([1,0])
                    else:
                        lable_class.append([0,1])
                # image = handle_sparse(image,batch_size)
                yield inputs, predicts, lable_class
                i+=1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    print("hello,world")
    exit(0)
    for inputs, predicts, label in batch_iter(r"../data/tf.records.train.positive.neg.txt",3,7):
        print(inputs)
        print(predicts)
        print(label)
