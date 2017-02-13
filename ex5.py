# coding:utf-8

import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd
from numpy.random import randint

df = pd.read_csv("./data_csv/ex4.csv", names=('in', 'x1', 'x2', 't1', 't2', 't3'), index_col='in')

num_data = int(len(df)*0.8)

train_set = df[:num_data]
test_set = df[num_data:]

train_x = train_set[['x1', 'x2']].as_matrix()
train_t = train_set[['t1', 't2', 't3']].as_matrix().reshape([len(train_set), 3])
test_x = test_set[['x1', 'x2']].as_matrix()
test_t = test_set[['t1', 't2', 't3']].as_matrix().reshape([len(test_set), 3])

x = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.zeros([2, 3]))
w0 = tf.Variable(tf.zeros([3]))
f = tf.matmul(x, w) + w0
p = tf.nn.softmax(f)

t = tf.placeholder(tf.float32, [None, 3])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(4000):
    i += 1
    target = randint(0, 80, 10)
    run_x = train_x[target,]
    run_t = train_t[target,]

    sess.run(train_step, feed_dict={x: run_x, t: run_t})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x:test_x, t:test_t})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

w0_val, w_val = sess.run([w0, w])
print w0_val
print w_val