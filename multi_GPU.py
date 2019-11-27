import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import numpy as np
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#### id of gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#### training data
#### shape=(n_sample, n_code=854)
REAL = np.load('')

#### demographic for training data
#### shape=(n_sample, 6)
#### if sample_x is male, then LABEL[x,0]=1, else LABEL[x,1]=1
#### if sample_x's is within 0-17, then LABEL[x,2]=1
#### elif sample_x's is within 18-44, then LABEL[x,3]=1
#### elif sample_x's is within 45-64, then LABEL[x,4]=1
#### elif sample_x's is within 64-, then LABEL[x,5]=1
LABEL = np.load('')

#### training parameters
NUM_GPUS = 1
BATCHSIZE_PER_GPU = 2000
TOTAL_BATCHSIZE = BATCHSIZE_PER_GPU * NUM_GPUS
STEPS_PER_EPOCH = int(np.load('ICD9/train.npy').shape[0] / 2000)

g_structure = [128, 128]
d_structure = [854, 256, 128]
z_dim = 128

def _variable_on_cpu(name, shape, initializer=None):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def batchnorm(inputs, name, labels=None, n_labels=None):
    mean, var = tf.nn.moments(inputs, [0], keep_dims=True)
    shape = mean.shape[1].value
    offset_m = _variable_on_cpu(shape=[n_labels,shape], name='offset'+name,
                                initializer=tf.zeros_initializer)
    scale_m = _variable_on_cpu(shape=[n_labels,shape], name='scale'+name,
                               initializer=tf.ones_initializer)
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-8)
    return result


def layernorm(inputs, name, labels=None, n_labels=None):
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    shape = inputs.shape[1].value
    offset_m = _variable_on_cpu(shape=[n_labels,shape], name='offset'+name,
                                initializer=tf.zeros_initializer)
    scale_m = _variable_on_cpu(shape=[n_labels,shape], name='scale'+name,
                               initializer=tf.ones_initializer)
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-8)
    return result


def input_fn():
    features_placeholder = tf.placeholder(shape=REAL.shape, dtype=tf.float32)
    labels_placeholder = tf.placeholder(shape=LABEL.shape, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.repeat(10000)
    dataset = dataset.batch(batch_size=BATCHSIZE_PER_GPU)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # init_op = iterator.initializer
    return iterator, features_placeholder, labels_placeholder


def generator(z, label):
    x = z
    tmp_dim = z_dim
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE, regularizer=l2_regularizer(0.00001)):
        for i, dim in enumerate(g_structure[:-1]):
            kernel = _variable_on_cpu('W_' + str(i), shape=[tmp_dim, dim])
            h1 = batchnorm(tf.matmul(x, kernel), name='cbn' + str(i), labels=label, n_labels=8)
            h2 = tf.nn.relu(h1)
            x = x + h2
            tmp_dim = dim
        i = len(g_structure) - 1
        kernel = _variable_on_cpu('W_' + str(i), shape=[tmp_dim, g_structure[-1]])
        h1 = batchnorm(tf.matmul(x, kernel), name='cbn' + str(i),
                       labels=label, n_labels=8)
        h2 = tf.nn.tanh(h1)
        x = x + h2

        kernel = _variable_on_cpu('W_' + str(i+1), shape=[128, 854])
        bias = _variable_on_cpu('b_' + str(i+1), shape=[854])
        x = tf.nn.sigmoid(tf.add(tf.matmul(x, kernel), bias))
    return x


def discriminator(x, label):
    with tf.variable_scope('D', reuse=tf.AUTO_REUSE, regularizer=l2_regularizer(0.00001)):
        for i, dim in enumerate(d_structure[1:]):
            kernel = _variable_on_cpu('W_' + str(i), shape=[d_structure[i], dim])
            bias = _variable_on_cpu('b_' + str(i), shape=[dim])
            x = tf.nn.relu(tf.add(tf.matmul(x, kernel), bias))
            x = layernorm(x, name='cln' + str(i), labels=label, n_labels=8)
        i = len(d_structure)
        kernel = _variable_on_cpu('W_' + str(i), shape=[d_structure[-1], 1])
        bias = _variable_on_cpu('b_' + str(i), shape=[1])
        y = tf.add(tf.matmul(x, kernel), bias)
    return y


def compute_dloss(real, fake, label):
    epsilon = tf.random_uniform(
        shape=[BATCHSIZE_PER_GPU, 1],
        minval=0.,
        maxval=1.)
    x_hat = real + epsilon * (fake - real)
    y_hat_fake = discriminator(fake, label)
    y_hat_real = discriminator(real, label)
    y_hat = discriminator(x_hat, label)

    grad = tf.gradients(y_hat, [x_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), 1))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    w_distance = -tf.reduce_mean(y_hat_real) + tf.reduce_mean(y_hat_fake)
    loss = w_distance + 10 * gradient_penalty + sum(all_regs)
    tf.add_to_collection('dlosses', loss)

    return w_distance, loss


def compute_gloss(fake, label):
    y_hat_fake = discriminator(fake, label)
    all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = -tf.reduce_mean(y_hat_fake) + sum(all_regs)
    tf.add_to_collection('glosses', loss)
    return loss, loss


def tower_loss(scope, stage, real, label):
    label = tf.cast(label, tf.int32)
    label = label[:, 1] * 4 + tf.squeeze(
        tf.matmul(label[:, 2:], tf.constant([[0], [1], [2], [3]], dtype=tf.int32)))
    z = tf.random_normal(shape=[BATCHSIZE_PER_GPU, z_dim])
    fake = generator(z, label)
    if stage == 'D':
        w, loss = compute_dloss(real, fake, label)
        losses = tf.get_collection('dlosses', scope)
    else:
        w, loss = compute_gloss(fake, label)
        losses = tf.get_collection('glosses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # loss_averages_op = loss_averages.apply(losses + [total_loss])
    #
    # with tf.control_dependencies([loss_averages_op]):
    #     total_loss = tf.identity(total_loss)

    return total_loss, w


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def graph(stage, opt):
    # global_step = tf.get_variable(stage+'_step', [], initializer=tf.constant_initializer(0), trainable=False)
    tower_grads = []
    per_gpu_w = []
    iterator, features_placeholder, labels_placeholder = input_fn()
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('TOWER', i)) as scope:
                    (real, label) = iterator.get_next()
                    loss, w = tower_loss(scope, stage, real, label)
                    tf.get_variable_scope().reuse_variables()
                    vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=stage)
                    grads = opt.compute_gradients(loss, vars_)
                    tower_grads.append(grads)
                    per_gpu_w.append(w)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads)

    mean_w = tf.reduce_mean(per_gpu_w)
    train_op = apply_gradient_op
    return train_op, mean_w, iterator, features_placeholder, labels_placeholder


def train(max_epochs, train_dir):
    with tf.device('/cpu:0'):
        opt_d = tf.train.AdamOptimizer(1e-4)
        opt_g = tf.train.AdamOptimizer(1e-4)
        train_d, w_distance, iterator_d, features_placeholder_d, labels_placeholder_d = graph('D', opt_d)
        train_g, _, iterator_g, features_placeholder_g, labels_placeholder_g = graph('G', opt_g)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(init)
            sess.run(iterator_d.initializer,
                     feed_dict={features_placeholder_d: REAL, labels_placeholder_d: LABEL})
            sess.run(iterator_g.initializer,
                     feed_dict={features_placeholder_g: REAL, labels_placeholder_g: LABEL})

            for epoch in range(1, max_epochs + 1):
                start_time = time.time()
                w_sum = 0
                for i in range(STEPS_PER_EPOCH):
                    for _ in range(2):
                        _, w = sess.run([train_d, w_distance])
                        w_sum += w
                    sess.run(train_g)
                duration = time.time() - start_time

                assert not np.isnan(w_sum), 'Model diverged with loss = NaN'

                format_str = 'epoch: %d, w_distance = %f (%.1f)'
                print(format_str % (epoch, -w_sum/(STEPS_PER_EPOCH*2), duration))
                if epoch % 500 == 0:
                    # checkpoint_path = os.path.join(train_dir, 'multi')
                    saver.save(sess, train_dir, write_meta_graph=False, global_step=epoch)
                    # saver.save(sess, train_dir, global_step=epoch)


def generate(model_dir, synthetic_dir, demo):
    tf.reset_default_graph()
    z = tf.random_normal(shape=[BATCHSIZE_PER_GPU, z_dim])
    y = tf.placeholder(shape=[BATCHSIZE_PER_GPU, 6], dtype=tf.int32)
    label = y[:, 1] * 4 + tf.squeeze(tf.matmul(y[:, 2:], tf.constant([[0], [1], [2], [3]], dtype=tf.int32)))
    fake = generator(z, label)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_dir)
        for m in range(2):
            for n in range(2, 6):
                idx1 = (demo[:, m] == 1)
                idx2 = (demo[:, n] == 1)
                idx = [idx1[j] and idx2[j] for j in range(len(idx1))]
                num = np.sum(idx)
                nbatch = int(np.ceil(num / BATCHSIZE_PER_GPU))
                label_input = np.zeros((nbatch*BATCHSIZE_PER_GPU, 6))
                label_input[:, n] = 1
                label_input[:, m] = 1
                output = []
                for i in range(nbatch):
                    f = sess.run(fake,feed_dict={y: label_input[i*BATCHSIZE_PER_GPU:(i+1)*BATCHSIZE_PER_GPU]})
                    output.extend(np.round(f))
                output = np.array(output)[:num]
                np.save(synthetic_dir + str(m) + str(n), output)


if __name__ == '__main__':
    #### args_1: number of training epochs
    #### args_2: dir to save the trained model
    train(500, '')

    #### args_1: dir of trained model
    #### args_2: dir to save synthetic data
    #### args_3, label of data-to-be-generated
    generate('', '', demo=LABEL)

