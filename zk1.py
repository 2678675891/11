import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler  # 特征缩放

# 鸢尾花
# 1.现有题目如下:
# (1)题目分析:
# ①正确导入相关头文件的包。
tf.set_random_seed(666)
# ②通过对以上关于Iris数据的分析，自动加载。
data = load_iris()
x = data.data
y = data.target

y_dim = len(set(y))
# ③对加载的数据及进行特征缩放
x = StandardScaler().fit_transform(x)
# ④定义tf.placeholder，对label进行one-hot处理。
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.int32, shape=[None, ])
Y_one = tf.one_hot(indices=Y, depth=y_dim)
# ⑤加入两层隐藏层，层数自行设计。
# ⑥根据网络模型结构,设置每一层weight,bias。
w1 = tf.Variable(tf.random_normal(shape=[4, 128]))
b1 = tf.Variable(tf.random_normal(shape=[128, ]))
h1 = tf.sigmoid(tf.matmul(X, w1) + b1)

w2 = tf.Variable(tf.random_normal(shape=[128, y_dim]))
b2 = tf.Variable(tf.random_normal(shape=[y_dim, ]))
h2 = tf.matmul(h1, w2) + b2
# ⑦写出loss函数(交叉嫡损失函数)，计算损失。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=Y_one))
op = tf.train.AdamOptimizer(0.01).minimize(loss)
# ⑧写出accuracy计算逻辑,计算精度
y_true = tf.argmax(Y_one, axis=-1)
y_predict = tf.argmax(h2, axis=-1)
acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_predict), tf.float32))
# ⑨每100次打印步数,损失和精度;
# 10对关键步骤给出注解
# 11代码逻辑过程清晰
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        loss_, op_, acc_ = sess.run([loss, op, acc], feed_dict={X: x, Y: y})
        if i % 100 == 0:
            print(i, loss_)
