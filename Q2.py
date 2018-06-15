import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SHAPE = [-1, 28, 28, 1]
POST_CONV_SIZE = [-1, 7 * 7 * 64]


def create_conv_layer(x, row, col, filters):
    conv = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=[row, col],
                            padding="same", activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    return pool


def create_dense_layer(x, size, output=False):
    if output:
        dense = tf.layers.dense(inputs=x, units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

    else:
        dense = tf.layers.dense(inputs=x, units=size, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

    return dense


def create_cnn_net(input_layer):
    conv1 = create_conv_layer(input_layer, 5, 5, 32)
    conv2 = create_conv_layer(conv1, 5, 5, 64)
    flattened = tf.reshape(conv2, POST_CONV_SIZE)
    dense1 = create_dense_layer(flattened, 1024)
    dense2 = create_dense_layer(dense1, 1024)
    out = create_dense_layer(dense2, 10, output=True)

    return out


def create_training_net(features, labels, mode):
    input_layer = tf.reshape(features, IMAGE_SHAPE)
    cnn_net = create_cnn_net(input_layer)

    predictions = tf.argmax(input=cnn_net, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=cnn_net)

    cost_logging =tf.train.LoggingTensorHook({"cost": loss},
                                             every_n_iter=250)

    if mode ==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimize = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=optimize,
                                          training_hooks=[cost_logging])

    accuracy_metric = {"accuracy": tf.metrics.accuracy(labels=labels,
                                                       predictions=predictions)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=accuracy_metric)


def main(unused_argv):
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  x_train = mnist.train.images  # Returns np.array
  y_train = np.asarray(mnist.train.labels, dtype=np.int32)
  x_test = mnist.test.images  # Returns np.array
  y_test = np.asarray(mnist.test.labels, dtype=np.int32)

  classifier = tf.estimator.Estimator(model_fn=create_training_net)

  train_net = tf.estimator.inputs.numpy_input_fn(x=x_train,
                                                 y=y_train,
                                                 batch_size=100,
                                                 num_epochs=None,
                                                 shuffle=True)
  classifier.train(input_fn=train_net, steps=5000)

  test_net = tf.estimator.inputs.numpy_input_fn(x=x_test,
                                                y=y_test,
                                                num_epochs=1,
                                                shuffle=False)
  results = classifier.evaluate(input_fn=test_net)
  print(results)


if __name__ == "__main__":
    tf.app.run()
