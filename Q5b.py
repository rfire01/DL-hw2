import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SHAPE = [-1, 28, 28, 1]
POST_CONV_SIZE = [-1, 7 * 7 * 64]


def create_conv_layer(x, row, col, filters, mode):
    conv = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=[row, col],
                            padding="same",
                            kernel_initializer=tf.contrib.layers.xavier_initializer())

    norm = tf.layers.batch_normalization(conv,
                                         training=mode == tf.estimator.ModeKeys.TRAIN)

    norm = tf.nn.relu(norm)

    pool = tf.layers.max_pooling2d(inputs=norm, pool_size=[2, 2], strides=2)

    return pool


def create_dense_layer(x, size, output=False):
    if output:
        dense = tf.layers.dense(inputs=x, units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

    else:
        dense = tf.layers.dense(inputs=x, units=size, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())

    return dense


def create_cnn_net(input_layer, mode):
    conv1 = create_conv_layer(input_layer, 5, 5, 32, mode)
    conv2 = create_conv_layer(conv1, 5, 5, 64, mode)
    flattened = tf.reshape(conv2, POST_CONV_SIZE)
    dense1 = create_dense_layer(flattened, 1024)
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = create_dense_layer(dropout, 1024)
    out = create_dense_layer(dense2, 10, output=True)

    return out


def create_training_net(features, labels, mode):
    input_layer = tf.reshape(features, IMAGE_SHAPE)
    cnn_net = create_cnn_net(input_layer, mode)

    predictions = tf.argmax(input=cnn_net, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=cnn_net)

    cost_logging =tf.train.LoggingTensorHook({"cost": loss},
                                             every_n_iter=250)

    if mode ==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
        optimize = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=optimize,
                                          training_hooks=[cost_logging])

    metrics = {"accuracy": tf.metrics.accuracy(labels=labels,
                                               predictions=predictions)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=metrics)


def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    last_loss = np.inf
    before_last_loss = np.inf

    x_train = mnist.train.images  # Returns np.array
    y_train = np.asarray(mnist.train.labels, dtype=np.int32)

    x_validation = mnist.validation.images
    y_validation = np.asarray(mnist.validation.labels, dtype=np.int32)

    x_test = mnist.test.images  # Returns np.array
    y_test = np.asarray(mnist.test.labels, dtype=np.int32)

    classifier = tf.estimator.Estimator(model_fn=create_training_net,)
                                      # config=tf.contrib.learn.RunConfig(save_checkpoints_steps=10))

    train_net = tf.estimator.inputs.numpy_input_fn(x=x_train,
                                                 y=y_train,
                                                 batch_size=100,
                                                 num_epochs=None,
                                                 shuffle=True)

    validation_net = tf.estimator.inputs.numpy_input_fn(x=x_validation,
                                                        y=y_validation,
                                                        num_epochs=1,
                                                        shuffle=False)

    for i in range(5000):
        classifier.train(input_fn=train_net, steps=1)
        validation_res = classifier.evaluate(input_fn=validation_net)
        print('int iteration ', i, ' loss is = ', validation_res['loss'])
        if before_last_loss <= last_loss <= validation_res['loss']:
            print('no improvement for 3 batches in a row')
            print('stopping training')
            break;

        before_last_loss = last_loss
        last_loss = validation_res['loss']

    test_net = tf.estimator.inputs.numpy_input_fn(x=x_test,
                                                y=y_test,
                                                num_epochs=1,
                                                shuffle=False)
    results = classifier.evaluate(input_fn=test_net)
    print(results)


if __name__ == "__main__":
    tf.app.run()
