# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, [4, 2])  # 変数 4組テストだから
w = tf.Variable(tf.zeros([2, 1]))  # 求めたい数
y = tf.matmul(x, w)
t = tf.placeholder(tf.float32, [4, 1])
loss = tf.reduce_sum(tf.square(y - t))

train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

train_t = np.array([[40],
                    [38],
                    [46],
                    [49]])

train_x = np.array([[1, 32],
                   [1, 23],
                   [1, 38],
                   [1, 48]])
i = 0
for _ in range(100000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 10000 == 0:
        loss_val = sess.run(loss, feed_dict={x:train_x, t:train_t})
        print ('Step: %d, Loss: %f' % (i, loss_val))


w_val = sess.run(w)
print w_val

def predict(x):
    result = 0.0
    for n in range(0, 2):
        result += w_val[n][0] * x**n
    return result


fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
subplot.set_xlim(1, 100)
subplot.scatter([32, 23, 38, 48], train_t)
linex = np.linspace(1, 100, 100)
liney = predict(linex)
subplot.plot(linex, liney)
