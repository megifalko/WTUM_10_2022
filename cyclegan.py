# Based on https://www.tensorflow.org/tutorials/generative/cyclegan and https://github.com/LynnHo/CycleGAN-Tensorflow-2

import tensorflow as tf
import os
import time
from IPython.display import clear_output
import argparse
from pathlib import Path
import module as m
import matplotlib.pyplot as plt

cwd = str(Path().absolute())

parser = argparse.ArgumentParser(description='Simple cycleGAN.')
parser.add_argument('--train', '-t', action='store_true', help='train the model', default=False)
parser.add_argument('--epochs', '-e', help='duration of training', default=10)
parser.add_argument('--count', '-c', help='number of images to generate', default=5)
parser.add_argument('--showProgress', '-s', action='store_true', help='show progress during training', default=False)
parser.add_argument('--path', '-p', help='path to training data, relative to current working directory', required=True)
args = parser.parse_args()

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

dataset = tf.keras.utils.image_dataset_from_directory(
  cwd + '\\' + args.path,
  labels='inferred', batch_size=None
)

testA = dataset.class_names.index('testA')
testB = dataset.class_names.index('testB')
trainA = dataset.class_names.index('trainA')
trainB = dataset.class_names.index('trainB')

train_B = dataset.filter(lambda img, label: label == trainB)
train_A = dataset.filter(lambda img, label: label == trainA)
test_B = dataset.filter(lambda img, label: label == testB)
test_A = dataset.filter(lambda img, label: label == testA)

def preprocess_image_train(image, label):
  image = m.random_jitter(image, IMG_WIDTH, IMG_HEIGHT)
  image = m.normalize(image)
  return image

def preprocess_image_test(image, label):
  image = m.normalize(image)
  return image

train_B = train_B.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_A = train_A.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_B = test_B.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_A = test_A.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_B = m.load('test.jpg')

generator_g = m.ResnetGenerator(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
generator_f = m.ResnetGenerator(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

discriminator_x = m.ConvDiscriminator(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
discriminator_y = m.ConvDiscriminator(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = cwd + "\\checkpoints\\" + os.path.basename(os.path.normpath(args.path)) + "\\train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored')

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = m.generator_loss(disc_fake_y)
    gen_f_loss = m.generator_loss(disc_fake_x)

    total_cycle_loss = m.calc_cycle_loss(real_x, cycled_x) + m.calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + m.identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + m.identity_loss(real_x, same_x)

    disc_x_loss = m.discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = m.discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

# Run the trained model on the test dataset
if(args.train == False):
  for inp in test_B.take(int(args.count)):
    m.generate_images(generator_g, inp, ckpt.save_counter.numpy() + 1)
else:
  for epoch in range(int(args.epochs)):
    print('Starting training for epoch {}'.format(epoch + 1))
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_B, train_A)):
      train_step(image_x, image_y)
      if n % 10 == 0:
        print ('.', end='', flush=True)
      n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_B) so that the progress of the model
    # is clearly visible.
    if(args.showProgress and (epoch+1) % 5 == 0):
      m.generate_images(generator_g, sample_B, ckpt.save_counter.numpy() + 1)

    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))