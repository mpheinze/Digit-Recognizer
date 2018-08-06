##### Python Test Script #####

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

from tensorflow.examples.tutorials.mnist import input_data
dataset = input_data.read_data_sets('./Digit-Recognizer/', one_hot=True)

#defining constants
num_pictures = len(dataset.train.labels)
features = len(dataset.train.images[0])
imageshape = (28,28)
num_classes = len(dataset.train.labels[0])

dataset.test.cls = dataset.test.labels.argmax(axis=0)

#Batch size for mini-batch gradient descent
batch_size = 1000

#Placeholder variables - input data going into the model
x = tf.placeholder(tf.float32, [None, features])
y = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

#Model variables
weights = tf.Variable(tf.zeros([features,num_classes]))
bias = tf.Variable(tf.zeros(num_classes))

#Model - what is done with the variables
logits = tf.matmul(x,weights)+bias
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred)

#Costfunction for the model 
costfunction = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y)
cost = tf.reduce_mean(costfunction)

#optimisation function
optimisation_function = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

#Starts a session and intialises global variables
session = tf.Session()
session.run(tf.global_variables_initializer())

feed_dict_test = {
    x: dataset.test.images,
    y: dataset.test.labels,
    y_true_cls: dataset.test.cls
}

def optimiser (num_iterations):
    for i in range(num_iterations):
        xbtach, ybatch = dataset.train.next_batch(batch_size)
        feed_dict_train = {
            x: xbtach,
            y: ybatch
        }
        session.run(optimisation_function, feed_dict=feed_dict_train)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.metrics.accuracy(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def accuracy_function ():
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("accuracy on test set: {0:.1%}".format(acc))

accuracy_function()
optimiser(1000)
accuracy_function()

def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = dataset.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

print_confusion_matrix()















