import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

num_classes = 10
num_features = 784

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train , x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255, x_test / 255

# def display_sample(num):
#     label = y_train[num]

#     image = x_train[num].reshape([28, 28])
#     plt.title('Sample: %d  Label: %d' % (num, label))
#     plt.imshow(image, cmap = plt.get_cmap('gray_r'))
#     plt.show()

# display_sample(800)

# images = x_train[0].reshape([1, 784])
# for i in range(1, 500):
#     images = np.concatenate((images, x_train[i].reshape([1, 784])))

# plt.imshow(images, cmap = plt.get_cmap('gray_r'))
# plt.show()

# training parameters
learning_rate = 0.01
training_steps = 5000
batch_size = 250
display_step = 100

n_hidden1 = 512
n_hidden2 = 256

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

random_normal = tf.initializers.RandomNormal()

weights = {
    'h1' : tf.Variable(random_normal([num_features, n_hidden1])),
    'h2' : tf.Variable(random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(random_normal([n_hidden2, num_classes]))
}

biases = {
    'b1' : tf.Variable(random_normal([n_hidden1])),
    'b2' : tf.Variable(random_normal([n_hidden2])),
    'out' : tf.Variable(random_normal([num_classes]))
}

def neural_net(inputData):
    hidden_layer1 = tf.add(tf.matmul(inputData, weights['h1']), biases['b1'])
    hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)
    hidden_layer2 = tf.nn.sigmoid(hidden_layer2)

    out_layer = tf.add(tf.matmul(hidden_layer2, weights['out']), biases['out'])

    return tf.nn.softmax(out_layer)

# loss function
def cross_entropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth = num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

# gradient descent optimizer

optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
    
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# accuracy metric

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

#  running training for given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Training epoch: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))
    
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# n_images = 200
# test_images = x_test[:n_images]
# test_labels = y_test[:n_images]
# predictions = neural_net(test_images)

# for i in range(n_images):
#     model_prediction = np.argmax(predictions.numpy()[i])
#     if(model_prediction != test_labels[i]):
#         plt.imshow(np.reshape(test_images[i], [28, 28]), cmap = 'gray_r')
#         plt.show()
#         print("Original Labels: %i" % test_labels[i])
#         print("Model prediction: %i" % model_prediction)
