import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util


BATCH_SIZE = 100
MAX_STEP = 3000

LEARNING_RATE = 0.8
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
        layer1 = tf.nn.relu(tf.matmul(input, weights) + bias, name='layer1')

        var_summary('weights', weights)
        var_summary('bias',bias)
        tf.summary.histogram('layer1_value', layer1)

    with tf.variable_scope('layer2'):
        weights = tf.get_variable('weights', [100, 50], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [50], initializer=tf.constant_initializer(0.))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + bias, name='layer2')

        var_summary('weights', weights)
        var_summary('bias', bias)
        tf.summary.histogram('layer2_value', layer2)

    with tf.variable_scope('layer3'):
        weights = tf.get_variable('weights', [50, 10], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.))
        layer3 = tf.add(tf.matmul(layer2, weights), bias, name='layer3')

        var_summary('weights', weights)
        var_summary('bias', bias)
        tf.summary.histogram('layer3_value', layer3)

    return layer3

def train(mnist):
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    global_step = tf.Variable(0, trainable=False)
    y = inference(x)

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y)
        loss = tf.reduce_mean(cross_entropy)
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

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init.run()
        writer = tf.summary.FileWriter('pb-path', sess.graph)
        for i in range(MAX_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, acc, step, summary = sess.run([train_step, loss, accuracy, global_step, merged], feed_dict={x: xs, y_:ys})

            if step%100 == 0:
                writer.add_summary(summary, step)

            if step%1000 == 0:
                print('{} epoches, loss: {}, accuracy: {}'.format(step, loss_value, acc))

        constant_graph = graph_util.convert_variables_to_constants(sess=sess,
                                                                   input_graph_def=sess.graph_def,
                                                                   output_node_names=['input/y-input', 'layer3/layer3', 'accuracy/accuracy'])

        with tf.gfile.FastGFile('pb-path/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def evaluate(mnist):
    # 导入pb文件到graph中
    with tf.gfile.FastGFile('pb-path/model.pb', mode='rb') as f:
        graph_def = tf.GraphDef()  # 复制定义好的计算图到新的图中，先创建一个空的图
        graph_def.ParseFromString(f.read())   # 加载proto-buf中的模型
        _ = tf.import_graph_def(graph_def, name='')   # 复制pre-def图的到默认图中

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x-input:0')
    y_ = graph.get_tensor_by_name('input/y-input:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

    with tf.Session() as sess:
        print('validation accuracy: {}'.format(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_:mnist.validation.labels})))

def fine_tuning(mnist):
    with tf.gfile.FastGFile('pb-path/model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input/x-input:0')
    y_ = graph.get_tensor_by_name('input/y-input:0')
    layer2 = graph.get_tensor_by_name('layer2/layer2:0')
    # layer2 = tf.stop_gradient(layer2)   # 如果需要freeze layer2和layer1层的参数，加上这名命令


    with tf.variable_scope('new-layer3'):
        weights = tf.get_variable('weights', [50, 20], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [20], initializer=tf.constant_initializer(0.))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + bias, name='layer3')
    with tf.variable_scope('layer4'):
        weights = tf.get_variable('weights', [20, 10], initializer=tf.truncated_normal_initializer(0.1))
        bias = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.))
        layer4 = tf.add(tf.matmul(layer3, weights), bias, name='layer4')

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=layer4)
        loss = tf.reduce_mean(cross_entropy, name='loss')

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.variable_scope('accuracy'):
        accuracy = tf.equal(tf.argmax(y_, 1), tf.argmax(layer4, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32), name='accuracy')


    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init.run()
        for i in range(2701):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, acc = sess.run([train_step, loss, accuracy], feed_dict={x:xs, y_:ys})
            if i%900 == 0:
                print('{} epoches, loss: {}, accuracy: {}'.format(i, loss_value, acc))
        constant_graph = graph_util.convert_variables_to_constants(sess=sess,
                                                                   input_graph_def=sess.graph_def,
                                                                   output_node_names=['input/y-input', 'layer4/layer4'])
        with tf.gfile.FastGFile('pb-path/fine_mode.pb', 'wb') as f:
            f.write(constant_graph.SerializeToString())

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(mnist)
    # evaluate(mnist)
    # fine_tuning(mnist)

if __name__ == '__main__':
    main()