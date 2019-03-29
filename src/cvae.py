import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import os
import cv2

in_dim = 28 * 28
h_dim = 100
z_dim = 128
c_dim = 10
test_num = 10
batch_size = 32
eps = 100
threshold = 0.1


def xavier_init(size):
    in_dim = size[0]
    t = 1.0 / np.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=t)


X = tf.placeholder(dtype=tf.float32, shape=(None, in_dim))
C = tf.placeholder(dtype=tf.float32, shape=(None, c_dim))

test_C = tf.placeholder(dtype=tf.float32, shape=(None, c_dim))
Z = tf.placeholder(dtype=tf.float32, shape=(None, z_dim))

E_w1 = tf.Variable(initial_value=xavier_init((in_dim + c_dim, h_dim)))
E_b1 = tf.Variable(initial_value=tf.zeros(shape=(h_dim,)))

E_mu_w = tf.Variable(initial_value=xavier_init(size=(h_dim, z_dim)))
E_mu_b = tf.Variable(initial_value=tf.zeros(shape=(z_dim,)))

E_log_var_w = tf.Variable(initial_value=xavier_init(size=(h_dim, z_dim)))
E_log_var_b = tf.Variable(initial_value=tf.zeros(shape=(z_dim,)))


def encoder(x):
    h = tf.nn.relu(tf.matmul(x, E_w1) + E_b1)
    h = tf.cast(h, tf.float32)
    mu = tf.matmul(h, E_mu_w) + E_mu_b
    log_var = tf.matmul(h, E_log_var_w) + E_log_var_b
    return mu, log_var


with tf.name_scope("encode"):
    X_temp = tf.concat([X, C], axis=1)
    mu, log_var = encoder(X_temp)
    kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(log_var) - log_var - 1.0, axis=1)


def sampling(mu, log_var):
    p = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * p


with tf.name_scope("sampling"):
    p = sampling(mu, log_var)
    p = tf.concat((p, C), axis=1)

D_w1 = tf.Variable(initial_value=xavier_init((z_dim + c_dim, h_dim)))
D_b1 = tf.Variable(initial_value=tf.zeros(shape=(h_dim,)))

D_w2 = tf.Variable(initial_value=xavier_init(size=(h_dim, in_dim)))
D_b2 = tf.Variable(initial_value=tf.zeros(shape=(in_dim,)))


def decoder(p):
    h = tf.nn.relu(tf.matmul(p, D_w1) + D_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_w2) + D_b2)


with tf.name_scope("decoder"):
    logits = decoder(p)
    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), axis=1)

with tf.name_scope("self_create"):
    _p = tf.concat((Z, test_C), axis=1)
    _logit = decoder(_p)

loss = tf.reduce_mean(kl_loss + recon_loss)
solver = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("../logs/cvae", sess.graph)
    saver = tf.train.Saver()
    (x_train, y_train), _ = mnist.load_data()
    x_train[x_train > 0] = 1
    x_train = x_train.reshape((-1, 28 * 28))
    print(x_train.shape)

    y_train = y_train.astype(np.int)
    y_train = y_train.reshape((-1, 1))
    ohe = OneHotEncoder()
    y_train_one_hot = ohe.fit_transform(y_train).toarray()

    valid_y = np.arange(0, 10)
    valid_y = valid_y.reshape((-1, 1))
    valid_y = ohe.fit_transform(valid_y).toarray()
    if not os.path.exists("../result"):
        os.mkdir("../result")

    for i in range(eps):
        loss_arr = []
        for j in range(int(len(y_train) / batch_size)):
            rand_index = np.random.choice(len(x_train), size=batch_size)
            _, _loss = sess.run([solver, loss],
                                feed_dict={X: x_train[rand_index], C: y_train_one_hot[rand_index]}
                                )
            loss_arr.append(_loss)
        print(np.mean(loss_arr))
        saver.save(sess, "../model/cvae/cvae.ckpt")
        imgs = sess.run([_logit, ], feed_dict={test_C: valid_y, Z: np.random.standard_normal(size=(test_num, z_dim))})

        imgs = np.reshape(imgs, (-1, 28, 28))
        imgs[imgs > threshold] = 255
        imgs = np.round(imgs)

        imgs = imgs.astype(np.uint8)
        imgs = imgs[:, :, :, np.newaxis]

        for j in range(test_num):
            if not os.path.exists("../result/" + str(i)):
                os.mkdir("../result/" + str(i))
            cv2.imwrite("../result/" + str(i) + "/" + str(j) + ".jpg", imgs[j])
