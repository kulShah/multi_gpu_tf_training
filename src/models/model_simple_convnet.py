import tensorflow as tf

# Build a CNN
def conv_net(x, n_classes, dropout, reuse, is_training):
  # Define a scope for resuing the variables
  with tf.variable_scope('ConvNet', reuse=reuse):
    # MNIST data input is 1D vec of 784 features (28 * 28 pixels)
    # reshapre to match image format [height * width * n_channels]
    # tensor input becomes 4D : [batch_size, height, width, n_channels]
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    # conv layer with 64 kernels, each kernel of size 5
    x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu)
    # max pooling (down sampling) with strides of 2 and kernel size 2
    x = tf.layers.max_pooling2d(x, 2, 2)

    # conv layer with 256 kernels, each kernel of size 3
    x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)
    # conv layer with 512 kernels, each kernel of size 3
    x = tf.layers.conv2d(x, 513, 3, activation=tf.nn.relu)
    # max pooling (down sampling) with strides of 2 and kernel size 2
    x = tf.layers.max_pooling2d(x, 2, 2)

    # flatten the data to 1D vec to feed it to fully connected layer
    x = tf.contrib.layers.flatten(x)

    # fully connected layer
    x = tf.layers.dense(x, 2048)
    # apply dropout (if is_training == False, dropout is not applied)
    x = tf.layers.dropout(x, rate=dropout, training=is_training)

    # Fully connected layer
    x = tf.layers.dense(x, 1024)
    # apply dropout (if is_training == False, dropout is not applied)
    x = tf.layers.dropout(x, rate=dropout, training=is_training)

    # output layer, class prediction
    out = tf.layers.dense(x, n_classes)

    # because 'softmax_cross_entropy_with_logits' loss already applies softmax,
    # we only apply softmax to the testing network
    out = tf.nn.softmax(out) if not is_training else out

  return out