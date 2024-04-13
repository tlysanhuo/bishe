import numpy as np
import tensorflow as tf

# 构建堆叠自编码模型
class StackedAutoencoder:
    def __init__(self, layer_sizes, learning_rate):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        self.W = []
        self.b = []

        for i in range(self.num_layers - 1):
            input_dim = layer_sizes[i]
            hidden_dim = layer_sizes[i + 1]

            self.W.append(tf.Variable(tf.random.normal([hidden_dim, input_dim])))
            self.b.append(tf.Variable(tf.random.normal([hidden_dim])))

    def encode(self, X):
        hidden_layer = X

        for i in range(self.num_layers - 1):
            weights = self.W[i]
            biases = self.b[i]
            hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, tf.transpose(weights)), biases))

        return hidden_layer

    def decode(self, X):
        output_layer = X

        for i in range(self.num_layers - 2, -1, -1):
            weights = tf.transpose(self.W[i])
            biases = self.b[i]
            output_layer = tf.nn.sigmoid(tf.add(tf.matmul(output_layer, weights), biases))

        return output_layer

    def train(self, X, num_epochs, batch_size):
        X = np.array(X)
        total_batches = int(X.shape[0] / batch_size)

        input_layer = tf.placeholder("float", [None, self.layer_sizes[0]])

        hidden_layer = self.encode(input_layer)
        output_layer = self.decode(hidden_layer)

        # 定义损失函数
        loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(output_layer, input_layer), 2))

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                avg_cost = 0.0

                for i in range(total_batches):
                    batch = X[i * batch_size: (i + 1) * batch_size]
                    _, l = sess.run([optimizer, loss], feed_dict={input_layer: batch})
                    avg_cost += l / total_batches

                print("Epoch:", epoch + 1, "cost =", "{:.5f}".format(avg_cost))

            self.W = sess.run(self.W)
            self.b = sess.run(self.b)

# 示例用法

input_data = [0.247585664,0.250704381,0.254055762,0.25760313,0.262139948,]  # 输入数据
input_data = np.array(input_data)
input_data = input_data.reshape(-1, 1)  # 将一维数组转换为二维数组
input_data = np.array(input_data)
layer_sizes = [input_data.shape[1], hidden_dim1, hidden_dim2]  # 每个层的维度
learning_rate = 0.001  # 学习率
num_epochs = 100  # 迭代次数
batch_size = 64  # 批大小

sae = StackedAutoencoder(layer_sizes, learning_rate)
sae.train(input_data, num_epochs, batch_size)