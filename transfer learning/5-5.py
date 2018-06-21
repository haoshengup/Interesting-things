import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MAX_STEP = 5000
BATCH_SIZE = 100

LEARNING_RATE = 0.0005
LEARNING_RATE_DECAY = 0.99

def var_summary(name, var):
    var_mean = tf.reduce_mean(var)
    var_std = tf.sqrt(tf.reduce_mean(var - var_mean))

    tf.summary.histogram(name, var)
    tf.summary.scalar(name + '/mean', var_mean)
    tf.summary.scalar(name + '/std', var_std)

def inference(input):
    with tf.variable_scope('layer1'):
        weights = tf.get_variable('weights', [784, 100], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [100], initializer=tf.constant_initializer(0.))
        layer1 = tf.nn.relu(tf.matmul(input, weights) + bias, name='layer1_value')

        var_summary('weights', weights)
        var_summary('bias', bias)
        tf.summary.histogram('layer1', layer1)

    with tf.variable_scope('layer2'):
        weights = tf.get_variable('weights', [100, 50], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [50], initializer=tf.constant_initializer(0.))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + bias, name='layer2_value')

        var_summary('weights', weights)
        var_summary('bias', bias)
        tf.summary.histogram('layer2', layer2)

    with tf.variable_scope('layer3'):
        weights = tf.get_variable('weights', [50, 10], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.))
        layer3 = tf.add(tf.matmul(layer2, weights), bias, name='layer3_value')

        var_summary('weights', weights)
        var_summary('bias', bias)
        tf.summary.histogram('layer3', layer3)

    return layer3

def train(mnist):
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    y = inference(x)
    global_step = tf.Variable(0, trainable=False)


    with tf.variable_scope('loss'):
        cross_entroy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        loss = tf.reduce_mean(cross_entroy)

        tf.summary.scalar('loss_value', loss)

    with tf.variable_scope('train'):
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE,
                                                   global_step=global_step,
                                                   decay_steps=mnist.train.num_examples//BATCH_SIZE,
                                                   decay_rate=LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.variable_scope('accuracy'):
        accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        tf.summary.scalar('acuracy_value',accuracy)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('path-copy', sess.graph)
        init.run()
        for i in range(MAX_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, acc, summary, step = sess.run([train_step, loss, accuracy, merged, global_step], feed_dict={x: xs, y_: ys})

            if step%100 == 0:
                writer.add_summary(summary,step)

            if step%500 == 0:
                saver.save(sess, 'path-copy/model.ckpt', global_step)
                print('{} epoches, loss:{}, accuracy:{}'.format(step, loss_value, acc))

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()

