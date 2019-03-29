import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import cv2
import os

np.set_printoptions(threshold=1e6)

input_shape = (None, 28, 28)
output_shape = (-1, 28, 28)
h_dim = 100
in_dim = 28 * 28
z_dim = 100
test_num = 10
threshold = 0.1

x = tf.placeholder(dtype=tf.float32, shape=(None, in_dim))
z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim))


def he_xavier(size):
    in_dim = size[0]
    var = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=var)


E_w1 = tf.Variable(initial_value=he_xavier((in_dim, h_dim)))
E_b1 = tf.Variable(initial_value=tf.zeros(shape=[h_dim]))

E_mu_w = tf.Variable(initial_value=he_xavier((h_dim, z_dim)))
E_mu_b = tf.Variable(initial_value=tf.zeros(shape=[z_dim]))

E_log_var_w = tf.Variable(initial_value=he_xavier((h_dim, z_dim)))
E_log_var_b = tf.Variable(initial_value=tf.zeros(shape=[z_dim]))


def encoder(X):
    h = tf.nn.relu(tf.matmul(x, E_w1) + E_b1)

    mu = tf.matmul(h, E_mu_w) + E_mu_b
    var_log = tf.matmul(h, E_log_var_w) + E_log_var_b
    return mu, var_log


def sampling(mu, var_log):
    e = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(var_log / 2) * e


mu, log_var = encoder(x)
sample_x = sampling(mu, log_var)
kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1, axis=1)

D_w1 = tf.Variable(initial_value=he_xavier((z_dim, h_dim)))
D_b1 = tf.Variable(initial_value=tf.zeros(shape=[h_dim]))

D_w2 = tf.Variable(initial_value=he_xavier((h_dim, in_dim)))
D_b2 = tf.Variable(initial_value=tf.zeros(shape=[in_dim]))


def decoder(p):
    h = tf.nn.relu(tf.matmul(p, D_w1) + D_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_w2) + D_b2)


logit = decoder(sample_x)
m_logit = decoder(z)

recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logit), axis=1)
loss = tf.reduce_mean(recon_loss + kl_loss)
solver = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

batch_size = 64
epochs = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("../logs/vae", sess.graph)

    (x_train, y_train), _ = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28 * 28))
    (height, width) = x_train.shape
    # x_train = x_train / 256
    x_train[x_train > 0] = 1
    saver = tf.train.Saver()

    print(x_train[0])
    for i in range(epochs):
        loss_arr = []
        for j in range(0, int(len(x_train) / batch_size)):
            rand_index = np.random.choice(len(x_train), size=batch_size)
            _, _loss = sess.run(fetches=[solver, loss],
                                feed_dict={x: x_train[rand_index]})
            loss_arr.append(_loss)
        print(np.mean(loss_arr))
        saver.save(sess, "../model/model.ckpt")

        test_logit = sess.run(m_logit, feed_dict={z: np.random.standard_normal(size=(test_num, h_dim))})
        print(test_logit.shape)
        test_logit = np.reshape(test_logit, (test_num, 28, 28))
        test_logit = test_logit[:, :, :, np.newaxis]
        test_logit[test_logit > threshold] = 255
        test_logit = np.round(test_logit)
        test_logit = test_logit.astype(np.uint8)

        for j in range(test_num):
            if not os.path.exists("../model/" + str(i)):
                os.mkdir("../model/" + str(i))
            cv2.imwrite("../model/" + str(i) + "/" + str(j) + ".jpg", test_logit[j])
