import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
MAX_STEP = 3000

INPUT_NODE = 784
HIDDEN_NODE = 500
OUTPUT_NODE = 10

LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = 0.99

def var_summmary(name, var):
    var_mean = tf.reduce_mean(var)
    var_std = tf.sqrt(tf.reduce_mean(var - var_mean))

    tf.summary.histogram(name, var)
    tf.summary.scalar(name + '/mean', var_mean)
    tf.summary.scalar(name + '/std', var_std)

def inference(input):
    with tf.variable_scope('layer1'):
        weights = tf.get_variable('weights', [INPUT_NODE, HIDDEN_NODE], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [HIDDEN_NODE], initializer=tf.constant_initializer(0.))
        layer1 = tf.nn.relu(tf.matmul(input, weights) + bias, name='layer1')

        tf.summary.histogram('layer1_value', layer1)
        var_summmary('weights', weights)
        var_summmary('bias', bias)

    with tf.variable_scope('layer2'):
        weights = tf.get_variable('weights', [HIDDEN_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.))
        layer2 = tf.matmul(layer1, weights) + bias

        tf.summary.histogram('layer2_value', layer2)
        var_summmary('weights', weights)
        var_summmary('bias', bias)

    return layer2

def train(mnist):
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    global_step = tf.Variable(0, trainable=False)
    y = inference(x)

    with tf.variable_scope('loss'):
        cross_entroy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        loss = tf.reduce_mean(cross_entroy)
        tf.summary.scalar('loss_value', loss)

    with tf.variable_scope('accuracy'):
        accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy_value', accuracy)

    with tf.variable_scope('train'):
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE,
                                                   global_step=global_step,
                                                   decay_steps=mnist.train.num_examples//BATCH_SIZE,
                                                   decay_rate=LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        writer = tf.summary.FileWriter('path', sess.graph)

        for i in range(MAX_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, acc, summary, step = sess.run([train_step, loss, accuracy, merged, global_step], feed_dict={x: xs, y_:ys})

            if step%100 == 0:
                writer.add_summary(summary, step)

            if step%500 == 0:
                saver.save(sess, 'path/model.ckpt', global_step)
                print('{} epoches, loss: {}, accuracy: {}'.format(step, loss_value, acc))


def fine_tuning(mnist):
    saver = tf.train.import_meta_graph('path/model.ckpt-3000.meta')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x-input:0')
    y_ = graph.get_tensor_by_name('input/y-input:0')
    layer1 = graph.get_tensor_by_name('layer1/layer1:0')


    layer1 = tf.stop_gradient(layer1)   # 如果要freeze包括layer1之前的层，需要加上这行命令

    with tf.variable_scope('new-layer2'):
        weights = tf.get_variable('weights', [HIDDEN_NODE, 100], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [100], initializer=tf.constant_initializer(0.))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + bias)

    with tf.variable_scope('layer3'):
        weights = tf.get_variable('weights', [100, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(0.))
        bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.))
        layer3 = tf.matmul(layer2, weights) + bias

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=layer3)
        loss = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('accuracy'):
        accuracy = tf.equal(tf.argmax(y_, 1), tf.argmax(layer3, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))


    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    new_saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        saver.restore(sess, 'path/model.ckpt-3000')

        for i in range(9701):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, acc = sess.run([train_step, loss, accuracy], feed_dict={x:xs, y_:ys})

            if i%900 == 0:
                new_saver.save(sess, 'path/fine_model.ckpt',i)
                print('{} epoches, loss: {}, accuracy: {}'.format(i, loss_value, acc))

#####验证freeze layer1是否成功
        # xs, ys = mnist.train.next_batch(1)
        #
        # print('before:')
        # print(sess.run(layer1, feed_dict={x:xs, y_:ys}))
        #
        # sess.run([train_step], feed_dict={x: xs, y_: ys})
        #
        # print('after:')
        # print(sess.run(layer1, feed_dict={x: xs, y_: ys}))

def fine_evaluate(mnist):
    '''
    测试layer1之前的层的参数是否被保存，如果out的值不全为0，则说明被保存
    '''
    saver = tf.train.import_meta_graph('path/fine_model.ckpt-2700.meta')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x-input:0')
    y_ = graph.get_tensor_by_name('input/y-input:0')
    layer1 = graph.get_tensor_by_name('layer1/layer1:0')

    with tf.Session() as sess:
        saver.restore(sess, 'path/fine_model.ckpt-2700')
        xs, ys = mnist.train.next_batch(1)
        out = sess.run(layer1, feed_dict={x: xs})
        print(out)

def evaluate(mnist):
    saver = tf.train.import_meta_graph('path/model.ckpt-3000.meta')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x-input:0')
    y_ = graph.get_tensor_by_name('input/y-input:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

    with tf.Session() as sess:
        saver.restore(sess, 'path/model.ckpt-3000')
        acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_:mnist.validation.labels})
        print('validation accuracy: {}'.format(acc))


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # train(mnist)
    # evaluate(mnist)

    fine_tuning(mnist)
    # fine_evaluate(mnist)

if __name__ == '__main__':
    main()