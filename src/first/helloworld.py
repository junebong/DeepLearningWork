import tensorflow as tf
import os

# アラートを出さない為のログレベル設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

print(tf.__version__)
