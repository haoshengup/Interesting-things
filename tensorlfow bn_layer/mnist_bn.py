from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import argparse


def model(input, is_training):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training':is_training, 'decay':0.9}):
        conv1 = slim.conv2d(input, 16, kernel_size=3, scope='conv1')
        pool1 = slim.max_pool2d(conv1, kernel_size=2, scope='pool1')
        conv2 = slim.conv2d(pool1, 32, kernel_size=3, scope='conv2')
        pool2 = slim.max_pool2d(conv2, kernel_size=2, scope='pool2')
        flatten = slim.flatten(pool2, scope='flatten')
        fc1 = slim.fully_connected(flatten, 500, scope='fc1')
        dropout = slim.dropout(fc1, is_training=is_training)
        fc2 = slim.fully_connected(dropout, 10, activation_fn=None, scope='out')
        return fc2

def train(mnist):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    global_step = tf.Variable(0, trainable=False)

    y = model(x, True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
    cross_entropy = tf.reduce_mean(cross_entropy)

    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate, global_step=global_step,
                                               decay_steps=mnist.train.num_examples // args.batch_size,
                                               decay_rate=args.learning_rate_decay)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        for i in range(args.max_step):
            xs, ys = mnist.train.next_batch(args.batch_size)
            xs = np.reshape(xs, [args.batch_size, 28, 28, 1])
            _, loss, acc, step = sess.run([train_step, cross_entropy, accuracy, global_step], feed_dict={x: xs, y_: ys})
            if step % 500 == 0:
                print('{} epoches, loss: {}, accuracy: {}'.format(step, loss, acc))

            if step % 1000 == 0:
                saver.save(sess, args.logs + 'mnist_bn_model', global_step)


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    y = model(x, False)

    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    feed_dict = {x: np.reshape(mnist.validation.images, [-1, 28, 28, 1]), y_: mnist.validation.labels}
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, args.logs + 'mnist_bn_model-' + str(args.max_step))
        acc = sess.run(accuracy, feed_dict=feed_dict)
        print('test accuracy: {}'.format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train or evaluate mnist of using bn layer')
    parser.add_argument('command', metavar='<command>', help='train or evaluate')
    parser.add_argument('--batch_size', default=100, help='num pictures of one batch')
    parser.add_argument('--learning_rate', default=0.8, help='initial learning rate of the net')
    parser.add_argument('--learning_rate_decay', default=0.9, help='rate decay after one epoch')
    parser.add_argument('--max_step',default=3000, help='total step for training')
    parser.add_argument('--logs', default='path/logs/', help='Logs and checkpoints directory')

    args = parser.parse_args()
    print('command:', args.command)
    print('batch_size:', args.batch_size)
    print('learning_rate:', args.learning_rate)
    print('learning_rate_decay:', args.learning_rate_decay)
    print('max_step:', args.max_step)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    if args.command == 'train':
        train(mnist)
    else:
        print('loading weights...')
        evaluate(mnist)