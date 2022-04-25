import tensorflow as tf
import time
import os
import argparse
import numpy as np


class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_variable("bias",
                                      shape=[self.num_outputs])

    def call(self, x, y):
        return x * y + self.bias


class Generator(tf.keras.Model):
    def __init__(self, n):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu) for dim in G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5, center=False, scale=False)] + \
                                 [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in G_DIMS[1:-1]]
        self.output_layer_code = tf.keras.layers.Dense(G_DIMS[-1], activation=tf.nn.sigmoid)
        self.pointwiselayer = PointWiseLayer(G_DIMS[0])
        self.embedding = tf.keras.layers.Embedding(n, 384)

    def call(self, x, category):
        h = self.dense_layers[0](x)
        x = self.pointwiselayer(self.batch_norm_layers[0](h), self.embedding(category))
        for i in range(1, len(G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = self.batch_norm_layers[i](h)
            x += h
        x = self.output_layer_code(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, n):
        super(Discriminator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in D_DIMS]
        self.embedding = tf.keras.layers.Embedding(n, 384)
        self.linear = tf.keras.layers.Dense(384)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, category):
        x = self.layer_norm_layers[0](self.dense_layers[0](x))
        for i in range(1, len(D_DIMS)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        c = self.linear(x)
        x_vec = c / tf.math.sqrt(tf.reduce_sum(c ** 2, axis=-1, keepdims=True))
        category = self.embedding(category)
        category = category / tf.math.sqrt(tf.reduce_sum(category ** 2, axis=-1, keepdims=True))
        x = self.output_layer(x) + tf.reduce_sum(category * x_vec, axis=-1, keepdims=True)
        return x


def train():
    data = np.load('data.npy').astype('float32')
    if args.num_labels != '0':
        labels = np.load('labels.npy').astype('int32')
        num_labels = int(args.num_labels)
    else:
        labels = np.zeros((data.shape[0], 1))
        num_labels = 1
    dataset_train = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(10000,
                                                                               reshuffle_each_iteration=True).batch(
        BATCHSIZE, drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)

    generator = Generator(num_labels)
    discriminator = Discriminator(num_labels)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator,
                                     discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)

    @tf.function
    def d_step(real, category):
        z = tf.random.normal(shape=[category.shape[0], Z_DIM])

        epsilon = tf.random.uniform(
            shape=[category.shape[0], 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, category)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real, category)
            fake_output = discriminator(synthetic, category)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch([interpolate])
                interpolate_output = discriminator(interpolate, category)
            w_grad = t.gradient(interpolate_output, [interpolate])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            disc_loss = 10 * gradient_penalty + w_distance

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step(category):
        z = tf.random.normal(shape=[category.shape[0], Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z, category)

            fake_output = discriminator(synthetic, category)

            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        code, category = batch
        category = tf.squeeze(category)
        disc_loss, w_distance = d_step(code, category)
        g_step(category)
        return disc_loss, w_distance

    print('training start')
    for epoch in range(50000):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        step = 0.0
        for batch_sample in dataset_train:
            loss, w = train_step(batch_sample)
            total_loss += loss
            total_w += w
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
        if epoch % 250 == 249:
            print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch))
        if epoch % 1000 == 999:
            checkpoint.save(file_prefix=checkpoint_prefix)


def gen(epoch, model):
    data = np.load('data.npy')
    if args.num_labels != '0':
        labels = tf.data.Dataset.from_tensor_slices(np.load('labels.npy').astype('int32')).batch(
            BATCHSIZE, drop_remainder=False)
        num_labels = int(args.num_labels)
    else:
        labels = tf.data.Dataset.from_tensor_slices(np.zeros((data.shape[0], 1), dtype='int32')).batch(
            BATCHSIZE, drop_remainder=False)
        num_labels = 1
    generator = Generator(num_labels)

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_prefix + '-' + str(epoch)).expect_partial()

    @tf.function
    def g_step(cat):
        z = tf.random.normal(shape=[1000, Z_DIM])
        synthetic = generator(z, cat)
        return synthetic

    syn = []
    for label in labels:
        syn.extend(g_step(label).numpy())
    np.save('synthetic' + str(model) + '_epoch' + str(epoch), syn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    parser.add_argument('num_labels', type=str)
    # number of labels, if zero, then non-label version will be used
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    checkpoint_directory = "training_checkpoints_emrwgan"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    BATCHSIZE = 1000
    Z_DIM = 128
    G_DIMS = [384, 384, 384, 384, 384, 384, 2591]
    D_DIMS = [384, 384, 384, 384, 384, 384]
    train()
