# coding:utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

df = pd.read_csv("../../data_csv/ex4.csv", names=('in', 'x1', 'x2', 't1', 't2', 't3'), index_col='in')

num_data = int(len(df)*0.8)

train_set = df[:num_data]
test_set = df[num_data:]

train_x = train_set[['x1', 'x2']].as_matrix()
train_t = train_set[['t1', 't2', 't3']].as_matrix().reshape([len(train_set), 3])
test_x = test_set[['x1', 'x2']].as_matrix()
test_t = test_set[['t1', 't2', 't3']].as_matrix().reshape([len(test_set), 3])

num_units = 3
mult = train_x.flatten().mean()

x = tf.placeholder(tf.float32, [None, 2])

w0 = tf.Variable(tf.truncated_normal([2, num_units]))
b0 = tf.Variable(tf.zeros([num_units]))
p = tf.nn.sigmoid(tf.matmul(x, w0) + b0 * mult)

t = tf.placeholder(tf.float32, [None, 3])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_rediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_rediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(1000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:train_x, t:train_t})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

train_set1 = train_set[train_set['t1']==1]
train_set2 = train_set[train_set['t2']==1]
train_set3 = train_set[train_set['t3']==1]

fig = plt.figure(figsize=(6,6))
subplot = fig.add_subplot(1,1,1)
subplot.set_ylim([0,6])
subplot.set_xlim([0,6])

subplot.scatter(train_set1.x1, train_set1.x2, marker='x')
subplot.scatter(train_set2.x1, train_set2.x2, marker='o')
subplot.scatter(train_set3.x1, train_set3.x2, marker='v')

locations = []

for x2 in np.linspace(0, 30, 100):
    for x1 in np.linspace(0, 30, 100):
        locations.append((x1, x2))

p_vals = sess.run(p, feed_dict={x:locations})
print(p_vals)
p_vals = p_vals.reshape((300,100))
subplot.imshow(p_vals, origin='lower', extent=(0,30,0,30), cmap=plt.cm.gray_r, alpha=0.5)

plt.show()