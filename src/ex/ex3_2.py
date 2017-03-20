# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../../data_csv/ex3.csv", names=('in', 'x1', 'x2', 't'), index_col='in')

num_data = int(len(df)*0.8)

train_set = df[:num_data]
test_set = df[num_data:]

train_x = train_set[['x1', 'x2']].as_matrix()
train_t = train_set['t'].as_matrix().reshape([len(train_set), 1])
test_x = test_set[['x1', 'x2']].as_matrix()
test_t = test_set['t'].as_matrix().reshape([len(test_set), 1])

# 関数数？
num_units = 1
# 平均値
mult = train_x.flatten().mean()

x = tf.placeholder(tf.float32, [None, 2])

w0 = tf.Variable(tf.zeros([2, num_units]))

b0 = tf.Variable(tf.zeros([num_units]))

p = tf.nn.sigmoid(tf.matmul(x, w0) + b0*mult)

t = tf.placeholder(tf.float32, [None, 1])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

i = 0
train_accuracy = []
test_accuracy = []
for _ in range(2000):
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    acc_val = sess.run(accuracy, feed_dict={x:train_x, t:train_t})
    train_accuracy.append(acc_val)
    acc_val = sess.run(accuracy, feed_dict={x:test_x, t:test_t})
    test_accuracy.append(acc_val)

w0_val, w_val = sess.run([b0, w0])
print(w0_val)
print(w_val)

fig = plt.figure(figsize=(8, 6))
subplot = fig.add_subplot(1, 1, 1)
subplot.plot(range(len(train_accuracy)), train_accuracy,
             linewidth=2, label='Training set')
subplot.plot(range(len(test_accuracy)), test_accuracy,
             linewidth=2, label='Test set')
subplot.legend(loc='upper left')

plt.show()