# coding:utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../data_csv/ex5.csv", names=('calyx_len', 'calyx_wid', 'petal_len', 'petal_wid', 'setosa', 'versicolor', 'virginia'), header=0)

num_data = int(len(df)*0.8)

train_set = df[:num_data]
test_set = df[num_data:]

train_x = train_set[['calyx_len', 'calyx_wid', 'petal_len', 'petal_wid']].as_matrix()
train_t = train_set[['setosa', 'versicolor', 'virginia']].as_matrix().reshape([len(train_set), 3])
test_x = test_set[['calyx_len', 'calyx_wid', 'petal_len', 'petal_wid']].as_matrix()
test_t = test_set[['setosa', 'versicolor', 'virginia']].as_matrix().reshape([len(test_set), 3])

num_units = 4 # ? どう決めるか？　変数の数とニアリ
mult = train_x.flatten().mean()
# print (mult)

x = tf.placeholder(tf.float32, [None, 4]) # 4つの変数

# 隠れ層1
w1 = tf.Variable(tf.truncated_normal([4, num_units])) # 4つの変数
b1 = tf.Variable(tf.zeros(num_units))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1 * mult)

# 出力層
w0 = tf.Variable(tf.truncated_normal([num_units, 3])) # 3つの分類
b0 = tf.Variable(tf.zeros([3])) # 3つの分類
p = tf.nn.sigmoid(tf.matmul(hidden1, w0) + b0 * mult)

t = tf.placeholder(tf.float32, [None, 3]) # 3つの分類

# 固定
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_rediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_rediction, tf.float32))

sess = tf.InteractiveSession() # 途中すポップ可能
sess.run(tf.initialize_all_variables())

train_accuracy = []
test_accuracy = []

i = 0
for _ in range(2000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    acc_val = sess.run(accuracy, feed_dict={x:train_x, t:train_t})
    train_accuracy.append(acc_val)
    acc_val = sess.run(accuracy, feed_dict={x:test_x, t:test_t})
    test_accuracy.append(acc_val)

'''
b0_val, w0_val = sess.run([b0, w0])
b0_val, w01_val, w02_val = b0_val[0], w0_val[0][0], w0_val[1][0]
print(w0_val, w01_val, w02_val)
'''

fig = plt.figure(figsize=(8,6))
subplot = fig.add_subplot(1,1,1)
subplot.plot(range(len(train_accuracy)), train_accuracy,
             linewidth=2, label='Training set')
subplot.plot(range(len(test_accuracy)), test_accuracy,
             linewidth=2, label='Test set')
subplot.legend(loc='upper left')

plt.show()

