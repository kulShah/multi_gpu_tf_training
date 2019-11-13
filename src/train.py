import numpy as np
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

from models.model_simple_convnet import conv_net
from train_config import config
from utils import average_gradients, assign_to_device

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

# hyperparameters for training
num_gpus = config.TRAIN.num_gpus
num_steps = config.TRAIN.num_steps
learning_rate = config.TRAIN.lr_init
batch_size = config.TRAIN.batch_size
display_step = config.TRAIN.display_step

# model parameters
num_input = config.MODEL.num_input
num_classes = config.MODEL.num_classes
dropout = config.MODEL.dropout

# place all ops on CPU by default
with tf.device('/cpu:0'):
  tower_grads = []
  reuse_vars = False

  # tf Graph input
  X = tf.placeholder(tf.float32, [None, num_input])
  Y = tf.placeholder(tf.float32, [None, num_classes])

  # Loop over all GPUs and construct their own computation graph
  for i in range(num_gpus):
    with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

      # split data between GPUs
      _x = X[i * batch_size : (i+1) * batch_size]
      _y = Y[i * batch_size : (i+1) * batch_size]

      # Since dropout has different behavior at training and prediction time,
      # we need to create 2 distinct computation graphs that share the same weights

      # create a graph for training
      logits_train = conv_net(_x, num_classes, dropout, reuse=reuse_vars, is_training=True)

      # create a graph for testing that resuse the same weights
      logits_test = conv_net(_x, num_classes, dropout, reuse=True, is_training=False)

      # define loss and optimizer (with train logits, for dropout to take effect)
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=_y))
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      grads = optimizer.compute_gradients(loss_op)

      # only first GPU compute accuracy
      if i == 0:
        # evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

      reuse_vars = True
      tower_grads.append(grads)

    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)

    # initializing the variables
    init = tf.global_variables_initializer()

    # launch the graph
    with tf.Session() as sess:

      # run the initializer
      sess.run(init)

      # step = 1

      # keep training until reach max iterations:
      for step in range(1, num_steps + 1):
        # get a batch for each GPU
        batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)

        # run optimization op (backprop)
        ts = time.time()
        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
        te = time.time() - ts

        if step % display_step == 0 or step == 1:
          # calculate batch loss and accuracy
          loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y:batch_y})
          print( "step = " + str(step) + \
                " minibatch loss = " + "{:.4f}".format(loss) + \
                " training accuracy = " + "{:.3f}".format(acc) + \
                " , %i examples/sec" % int(len(batch_x)/ te) )

        # step += 1
      print("optimization finished")

      # calculate accuracy for 1000 mnist test images
      print("testing accuracy = ", \
            np.mean( [ sess.run( accuracy, feed_dict={X: mnist.test.images[i:i+batch_size], Y: mnist.test.labels[i:i+batch_size] } ) \
                       for i in range(0, len(mnist.test.images), batch_size ) ] ) )