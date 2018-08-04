#author: jackqiu
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tfRecord import batch_iter
from average_onehot2 import AOModel

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("training_file_path", "train.data", " the file path of training data")
tf.flags.DEFINE_string("dev_file_path", "dev.data", " the file path of develop data")
tf.flags.DEFINE_string("embedding_file_path", "embedding.file", " the file path of embedding file data")
tf.flags.DEFINE_string("input_date", "20180405", "the date of the training data")
tf.flags.DEFINE_boolean("need_dev", False, "do we need to have a test on test data")
tf.flags.DEFINE_boolean("trainable", False, "do we need to traing the vid embedding information")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def my_batch_iter(file_name, zero_index_id, epoch=1):
    for _ in range(epoch):
        train_sequence = []
        length = []
        predict_labels = []
        # age sex items length predict_label
        with open(file_name, 'r') as reader:
            for line in reader:
                line = line.strip()
                tokens = line.split("\t")
                if len(tokens) != 5:
                    continue

                cur_items = [int(tokens[0]), int(tokens[1])]
                tt = tokens[2].split(" ")
                items = []
                for t in tt:
                    if t != "":
                        items.append(int(t))
                if len(items) == 0:
                    items.append(zero_index_id)
                length.append(len(items))
                if len(items) < 5:
                    items.extend([zero_index_id] * (5 - len(items)))
                cur_items.extend(items)
                predict_labels.append(int(tokens[4]))
                train_sequence.append(cur_items)
                if len(predict_labels) == 1024:
                    # print train_sequence, length, predict_labels
                    yield train_sequence, length, predict_labels
                    train_sequence = []
                    length = []
                    predict_labels = []


def dev_batch(file_name, zero_index_id):
    train_sequence = []
    length = []
    predict_labels = []
    # age sex items length predict_label
    with open(file_name, 'r') as reader:
        for line in reader:
            line = line.strip()
            tokens = line.split("\t")
            if len(tokens) != 5:
                continue

            cur_items = [int(tokens[0]), int(tokens[1])]
            tt = tokens[2].split(" ")
            items = []
            for t in tt:
                if t != "":
                    items.append(int(t))
            if len(items) == 0:
                items.append(zero_index_id)
            length.append(len(items))
            if len(items) < 5:
                items.extend([zero_index_id] * (5 - len(items)))
            cur_items.extend(items)
            predict_labels.append(int(tokens[4]))
            train_sequence.append(cur_items)
        return train_sequence, length, predict_labels


def read_embedding_file(filename):
    W = []
    index = 0
    with open(filename, 'r') as reader:
        for line in reader:
            line = line.strip()
            tokens = line.split("\t")
            W.append([float(x) for x in tokens[1].split(":")])
            index+=1
            if index%100000 ==0:
                print(index)
    return np.asarray(W)


with tf.Graph().as_default():
    # generate_train(r'D:\Data\tmp\output_with_neg.txt') #generate a fade training list
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    vocab_size = 2974077
    max_length = 7
    ########################################################
    age_dict = {}
    sex_dict = {}
    item_dict = {}
    reverse_item_dict = {}
    reverse_age_dict = {}
    reverse_sex_dict = {}
    with open("./data/age.dict", 'r') as reader:
        for line in reader:
            line = line.strip()
            tokens = line.split("\t")
            age_dict[tokens[0]] = int(tokens[1])
            reverse_age_dict[int(tokens[1])] = tokens[0]
    with open("./data/sex.dict", 'r') as reader:
        for line in reader:
            line = line.strip()
            tokens = line.split("\t")
            sex_dict[tokens[0]] = int(tokens[1])
            reverse_sex_dict[int(tokens[1])] = tokens[0]

    with open(FLAGS.embedding_file_path, 'r') as reader:
        index = 0
        for line in reader:
            line = line.strip()
            tokens = line.split("\t")
            item_dict[tokens[0]] = index
            reverse_item_dict[index] = tokens[0]
            index += 1
    vocab_size = len(reverse_item_dict) - 1
    pretrained_embeddings = read_embedding_file(FLAGS.embedding_file_path)
    # profile_info, sequence_length, num_classes, vocab_size, embedding_size, filter_size, num_filters, l2_reg_lambda = 0.0
    model = AOModel(age_vocab_size=13 + 1, sex_vocab_size=3 + 1, sequence_length=7, num_classes=3,
                    initial_embedding_file=FLAGS.embedding_file_path,
                    vocab_size=vocab_size,
                    embedding_size=100,
                    sex_dict=sex_dict,
                    age_dict=age_dict,
                    item_dict=item_dict,
                    filter_size=None, num_filters=None, l2_reg_lambda=0.01)

    # cdssm = TextCNN2(max_lengh,2,vocab_size,150,filter_size=3,num_filters=300,l2_reg_lambda=0.01)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)


    ##############################


    def dev_step(input_batch, predict_batch, item_lengthes, detail_input_s):
        feed_dict = {
            model.input_x_sequence: input_batch,
            model.input_c_sequence: predict_batch,
            model.input_item_lengths: item_lengthes,
            model.step: -1
            # cdssm.dropout_keep_prob: 1.0
        }

        W, loss, fco, exs, WWW, bbb, item_lengths, tmp_embed_x_sequence, dig_il, pcc = sess.run(
            [model.W, model.loss, model.fco, model.embed_x_sequence, model.WWW, model.bbb, model.input_item_lengths,
             model.tmp_embed_x_sequence,
             model.dig_il, model.pcc],
            feed_dict=feed_dict
            )

        for fc, dis, e in zip(fco, detail_input_s, exs):
            print("_________________________________")
            print("*************************************")
            word_sim = np.dot(W, fc)
            word_sim_argsort = word_sim.argsort()[::-1]
            print("\t".join([str(x) for x in dis]))
            print([(reverse_item_dict[i], word_sim[i]) for i in word_sim_argsort[0:10 + 1]])
        time_str = datetime.datetime.now().isoformat()
        print("develop {}:  loss {}".format(time_str, loss))


    def train_step(step, input_batch, predict_batch, item_lengthes, writer=None):
        """
        Evaluates model on a dev set
        """
        # print("begin ......")
        feed_dict = {
            model.input_x_sequence: input_batch,
            model.input_c_sequence: predict_batch,
            model.input_item_lengths: item_lengthes,
            model.step: step
            # cdssm.dropout_keep_prob: 1.0
        }

        _, step, summaries, loss, WWW, bbb, fco = sess.run(
            [train_op, global_step, train_summary_op, model.loss, model.WWW, model.bbb,
             model.fco],
            feed_dict=feed_dict
        )

        if step % 10000 == 0:
            with open("./model/www.model.%s.%d" % (FLAGS.input_date, step), 'w') as writer_WWW:
                for line in WWW:
                    output_line = " ".join([str(x) for x in line])
                    writer_WWW.write(output_line + "\n")
                b_str = " ".join([str(b) for b in bbb])
                writer_WWW.write(b_str + "\n")

        time_str = datetime.datetime.now().isoformat()
        if step % 100 == 0:
            print("{}: step {}, loss {}".format(time_str, step, loss))
        if writer:
            writer.add_summary(summaries, step)


    sess.run(tf.global_variables_initializer())
    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: pretrained_embeddings})
    i_step = 0
    dev_input_s, dev_lengths, dev_predict_labels = dev_batch(FLAGS.dev_file_path, vocab_size)
    detail_input_s = []
    for x in dev_input_s:
        tokens = x
        bf = []
        bf.append(reverse_age_dict[int(tokens[0])])
        bf.append(reverse_sex_dict[int(tokens[1])])
        for t in tokens[2:7]:
            if int(t) not in reverse_item_dict:
                continue
            bf.append(reverse_item_dict[int(t)])
        detail_input_s.append(bf)

    # for input_sequence, lengths, predict_labels in my_batch_iter("../data/traindata.average.shuffer.tail"):
    for input_sequence, lengths, predict_labels in my_batch_iter(FLAGS.training_file_path, vocab_size):
        train_step(i_step, input_batch=input_sequence, predict_batch=predict_labels, item_lengthes=lengths)
        i_step += 1
        current_step = tf.train.global_step(sess, global_step)
        if current_step % 100000 == 0 and FLAGS.need_dev:
            dev_step(input_batch=dev_input_s, item_lengthes=dev_lengths, predict_batch=dev_predict_labels,
                     detail_input_s=detail_input_s)

