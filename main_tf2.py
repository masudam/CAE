import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose
from tensorflow.keras import Model

# おまじない（自分の環境では必要）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class CAE(Model):
    def __init__(self):
        super(CAE,self).__init__()
        self.encoder = tf.keras.Sequential([
            Conv2D(16, 3, strides=(2,2), padding='same', activation=tf.nn.tanh), # 16x16x16
            Conv2D(32, 3, strides=(2,2), padding='same', activation=tf.nn.tanh), # 8x8x32
            Conv2D(32, 3, strides=(2,2), padding='same', activation=tf.nn.tanh), # 4x4x32
            Flatten(),
            Dense(4*4*8, activation=tf.nn.tanh), # 128 dim.
        ])
        self.decoder = tf.keras.Sequential([
            Dense(4*4*32, activation=tf.nn.tanh),
            Reshape((4,4,32)),
            Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.tanh), # 8x8x32
            Conv2DTranspose(16, 3, strides=2, padding='same', activation=tf.nn.tanh), # 16x16x16
            Conv2DTranspose(3, 3, strides=2, padding='same', activation=tf.nn.sigmoid), # 32x32x3
        ])

    def call(self, x):
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, z



# generate and save img
def generate_and_save_images(model, epoch, test_sample):
    predictions, _ = model(test_sample)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    plt.savefig('./result_tf2/image_at_epoch_{:04d}.png'.format(epoch))

# load cifar10 data in tf.keras.datasets
def prepare_data(train_size, test_size, batch_size):
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    def preprocess_images(images):
        images = images.reshape((images.shape[0], 32, 32, 3)) / 255.
        return images.astype('float32')

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                    .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Training Parameters
    train_size = 40000
    batch_size = 32
    test_size = 10000

    epochs = 30
    lr = 0.0001

    # prepare data
    train_dataset, test_dataset = prepare_data(train_size, test_size, batch_size)

    # save test sample img
    num_examples_to_generate = 16
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    fig = plt.figure(figsize=(4, 4))
    for i in range(test_sample.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(test_sample[i, :, :, :])
        plt.axis('off')
    plt.savefig('./result_tf2/test_sample.png')


    model = CAE()
    optimizer = tf.keras.optimizers.Adam(lr)

    generate_and_save_images(model, 0, test_sample)

    def calc_loss(model, x):
        x_pred, _ = model(x)
        loss = tf.reduce_mean(tf.pow(x-x_pred, 2))
        return loss

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            with tf.GradientTape() as tape:
                loss = calc_loss(model, train_x)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
        end_time = time.time()

        test_loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            test_loss(calc_loss(model, test_x))
        elbo = test_loss.result()
        print('Epoch: {}, Train set loss: {}, Test set loss: {}, time elapse for current epoch: {}'
                .format(epoch, loss, elbo, end_time - start_time))
        if epoch % 5 == 0:
            generate_and_save_images(model, epoch, test_sample)
