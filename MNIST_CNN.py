import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder(tf.float32,shape = [None, 784])
y= tf.placeholder(tf.float32,shape=[None,10])

BATCH_SIZE = 100
EPOCHS = 10

def weights(shape):
    return tf.get_variable("weights",shape=shape,initializer=tf.random_normal_initializer())

def biases(length):
    return tf.get_variable("biases",shape=[length],initializer=tf.random_normal_initializer())


def Conv2D(input, filter_shape, pool = "MAX", relu = "True"):
    # filter_shape = [height, width, in_channel, out_channel]
    w1 = weights(filter_shape)
    # num_biases = [out_channel]
    b1 = biases(filter_shape[3])
    conv = tf.nn.conv2d(input,w1,strides = [1,1,1,1],padding = "SAME")
    conv = conv + b1
    if(pool == "MAX"):
        conv = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    if(relu == "True"):
        conv = tf.nn.relu(features=conv)
    return conv

def FCLayer(input, n_nodes, num_out):
    fc = tf.reshape(input, [-1, n_nodes])
    w2 = weights([n_nodes, num_out])
    b2 = biases(num_out)
    output = tf.matmul(fc,w2) + b2
    return output

def Neural_Network(x):
    input = tf.reshape(x, [-1,28,28,1])
    with tf.variable_scope("C-1"):
        conv1 = Conv2D(input, [5,5,1,16])
    with tf.variable_scope("C-2"):
        conv2 = Conv2D(conv1,[3,3,16,32])
    with tf.variable_scope("FC-Layer"):
        fc_output = FCLayer(conv2, 1568, 10)
    return fc_output


prediction = Neural_Network(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = prediction))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_num in range(EPOCHS):
        epoch_loss = 0

        for _ in range(int(mnist.train.num_examples/100)):
            epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
            temp, loss = sess.run([optimizer, cost],feed_dict={x:epoch_x , y:epoch_y})
            epoch_loss += loss
            
        print("Epochs Completed: ", epoch_num, " out of ", EPOCHS,"        Loss of Epoch: ", epoch_loss)

    test_x = mnist.test.images
    test_y = mnist.test.labels
    accuracy_measure = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)),dtype=tf.float32))
    accuracy = sess.run(accuracy_measure,feed_dict={x:test_x, y:test_y})
    print("Accuracy of the Net: ",accuracy)

