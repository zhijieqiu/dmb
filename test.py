import tensorflow as tf


def my_self_attention(inputs, weights):
    input_T = tf.transpose(inputs, perm=[0, 2, 1])
    relation_matrix = tf.matmul(inputs, input_T)  # b*m*m
    related_sum = tf.reduce_sum(relation_matrix, axis=2)  # b*m

    weights_ = tf.reshape(tf.constant(weights), related_sum.get_shape().as_list)  # weights_ is b*m
    related_sum = tf.multiply(related_sum, weights_)  # related_sum is b*m
    exp_related_sum = tf.exp(related_sum)
    r_sum = tf.reduce_sum(exp_related_sum, axis=1, keep_dims=True)  # r_sum is b*1

    exp_related_sum_prob = tf.div(related_sum, r_sum)  # b*m
    exp_related_sum_prob = tf.expand_dims(exp_related_sum_prob, -1)

    outputs = tf.multiply(inputs, exp_related_sum_prob)
    attention_output = tf.reduce_sum(outputs, axis=1)
    return attention_output

def attention(x):
    x_T = tf.transpose(x,perm=[0,2,1]) #x -> b*m*n
    related_matrix = tf.matmul(x,x_T) #b*m*m
    related_sum = tf.reduce_sum(related_matrix,axis=2) #b*m
    exp_related_sum = tf.exp(related_sum)
    r_sum = tf.reduce_sum(exp_related_sum, axis= 1,keep_dims=True) #b*1
    exp_related_sum_prob = tf.div(exp_related_sum, r_sum)
    #related_sum = tf.div(related_sum,r_sum)
    outputs = tf.multiply(x,exp_related_sum_prob)
    return tf.reduce_sum(outputs,axis=1) #b*n



if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x = tf.constant([[1.0,2],[3,1]])
    x = tf.stack([x,x])

    a = sess.run(my_self_attention(x ,weights = [[1,1],[1,1]]))
    bxd = sess.run(a)
    print(bxd)
    exit(0)


    x_T = tf.transpose(x, perm=[0, 2, 1])  # x -> b*m*n

    related_matrix = tf.matmul(x, x_T)  # b*m*m

    related_sum = tf.reduce_sum(related_matrix, axis=2)  # b*m

    exp_related_sum = tf.exp(related_sum) #b*m



    r_sum = tf.reduce_sum(exp_related_sum, axis=1, keep_dims=True)  # b*1



    exp_related_sum_prob = tf.div(exp_related_sum, r_sum)
    # related_sum = tf.div(related_sum,r_sum)
    exp_related_sum_prob = tf.reshape(exp_related_sum_prob, [])
    outputs = tf.multiply(x, exp_related_sum_prob)
    final_outputs = tf.reduce_sum(outputs, axis=1)  # b*n

