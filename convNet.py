import matplotlib
matplotlib.use('Agg') # Allows us to save figure on the server
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.lines as mlines

'''
Implements a multi-layer convolutional neural network to recognize 24 mathematical symbols.
Adapted from Google's tensorflow tutorial

Architecture:
    First Convolution Layer:
        Convolution of 32 features for every 5x5 patch
        Max pooling 2x2
    Second Convolution Layer:
        Convolution of 64 features for every 5x5 patch
        Max pooling 2x2
    Fully Connected Layer:
        1024 fully connected neurons
    Dropout Layer:
        Randomly omits some outputs depending on dropout probability
    Output Layer:
        Final layer which will go through a softmax function
'''

def main():

    print("Starting up the Convolutional Neural Network!")
    print("beginning to import data")

    # Import training data
    # Used pandas to import data instead of numpy since it is much faster

    # Import training x data through pandas as a numpy ndarray
    trainX = pd.read_csv('data/train.csv', delimiter=' ', header=None).as_matrix()
    # trainX = np.loadtxt('data/train.csv', delimiter=' ')  # This was the old slower numpy method
    print("loaded training data X")

    # Import training y data through pandas as a numpy ndarray
    yTrainVals= pd.read_csv('data/trainLabel.csv', delimiter=' ', header=None).as_matrix()
    # yTrainVals = np.loadtxt('data/trainLabel.csv', delimiter=' ')  # This was the old slower numpy method
    #yTrainVals = yTrainVals.astype(int)
    print("loaded training data Y")
    yTrainVals = yTrainVals.flatten()  # Flatten the ndarray to a one-dimensional vector (needed in the next step)

    sizeInputVector = trainX.shape[1]  # Size of the input layer of the neural net and the input vectors
    sizeOutputVector = 24  # Size of the output layer of the neural net and the y-label vectors

    # Builds an array of y-label vectors for training data where the index of each 1 corresponds to the correct label
    trainY = np.zeros((len(trainX), sizeOutputVector))  # Filling with zeros
    trainY[np.arange(len(trainX)), yTrainVals-1] = 1  # Inserting a 1 at the location corresponding to the correct label
                                                # Subtracting 1 from yTrainVals because labels begin at 1 instead of 0


    # Import testing data
    # Used pandas to import data instead of numpy since it is much faster

    # Import testing x data through pandas as a numpy ndarray
    testX = pd.read_csv('data/test.csv', delimiter=' ', header=None).as_matrix()
    #testX = np.loadtxt('data/test.csv', delimiter=' ')
    print("loaded testing data X")

    # Import testing y data through pandas as a numpy ndarray
    yTestVals= pd.read_csv('data/testLabel.csv', delimiter=' ', header=None).as_matrix()
    #yTestVals = np.loadtxt('data/testLabel.csv', delimiter=' ')  # This was the old slower numpy method
    #yTestVals = yTestVals.astype(int)
    print("loaded testing data Y")
    yTestVals = yTestVals.flatten()  # Flatten the ndarray to a one-dimensional vector (needed in the next step)

    # Builds an array of y-label vectors for testing data where the index of each 1 corresponds to the label
    testY = np.zeros((len(testX), sizeOutputVector))  # Filling with zeros
    testY[np.arange(len(testX)), yTestVals-1] = 1  # Inserting a 1 at the location corresponding to the correct label
                                             # Subtracting 1 from yTrainVals because labels begin at 1 instead of 0


    # Create the Tensorflow model



    # x is a Tensorflow placeholder array. The first dimension is None, meaning it can expand as needed. The second
    # dimension is the size of the input vector to the neural network.
    x = tf.placeholder(tf.float32, [None, sizeInputVector])

    # This will hold the true labels of the inputs (truth-value output).
    y_ = tf.placeholder(tf.float32, [None, sizeOutputVector])


    # First Convolutional ayer

    # Weights of first convolutional layer. The first two dimensions are the patch size, the next is the number of input
    # channels, and the last is the number of output channels.
    W_conv1 = init_weights([5, 5, 1, 32])  # 32 features for each 5x5 patch
    b_conv1 = init_biases([32])  # Biases of first convolutional layer

    # Reshaping x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the
    # final dimension corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1, 35, 35, 1])

    # Convolve x_image with the weight tensor, add the bias, and apply the ReLU function
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # Apply max-pooling to reduce image size to 18x18



    # Second Convolutional Layer

    # Weights of second convolutional layer. The first two dimensions are the patch size, the next is the number of
    # input channels, and the last is the number of output channels.
    W_conv2 = init_weights([5, 5, 32, 64])  # 64 features for each 5x5 patch
    b_conv2 = init_biases([64])  # Biases of second convolutional layer

    # Convolve x_image with the weight tensor, add the bias, and apply the ReLU function
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # Apply max-pooling to reduce image size to 9x9



    # Densely Connected Layer

    # Weights of densely connected layer.
    W_fc1 = init_weights([9 * 9 * 64, 1024])  # 1024 neurons
    b_fc1 = init_biases([1024])    # Biases of densely connected layer

    # Reshape the tensor from the pooling layer into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 9 * 9 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Multiply by weight matrix, add bias, and apply ReLU.


    #Dropout

    # Placeholder for the probability that a neuron's output is kept during dropout.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # Automatically handles scaling neuron outputs


    # Readout Layer

    W_fc2 = init_weights([1024, sizeOutputVector])  # Weights of readout layer
    b_fc2 = init_biases([sizeOutputVector])  # Biases of readout layer

    # The output of the network is the dot product of the h_fc1_drop with the weights, W_fc2. This is added element-wise
    # to the biases, b_fc2.
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    # Using the cross entropy cost function averaged over all the samples in each batch. We apply the softmax function
    # on the unnormalized outputs to normalize them then compute the cost.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

    # Each training step is supposed to minimize the cross_entropy loss using the ADAM optimizer with a
    # learning rate of 0.5.
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # This launches the model in a Session
    sess = tf.InteractiveSession()

    # Checks whether or not the networks prediction, tf.argmax(y_conv, 1), matches the correct prediction, tf.argmax(y_, 1)
    # tf.argmax gives the index of the highest entry in a tensor along some axis
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    # To determine what fraction of the neural net's predictions are correct, we cast the correct_prediction vector to
    # floating point numbers and then take the mean.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())


    nEpochs = 100  # number of epochs to train the neural network on
    batchSize = 200  # Number of samples in each batch
    numBatches = int(len(trainX) / batchSize)  # Number of batches in each epoch

    trainAccuracies = [None] * nEpochs  # Vector to store the training accuracies
    testAccuracies = [None] * nEpochs  # Vector to store the testing accuracies

    print("beginning training")

    # Train the neural network
    for i in range(nEpochs):

        # Generate a random permutation of trainX and trainY while maintaining their relative order
        p = np.random.permutation(len(trainX))  # Generate a permutation of the numbers from 0 to len(trainX)
        xData = trainX[p]  # Shuffle trainX using the indices p
        yData = trainY[p]  # Shuffle trainY using the indices p

        # Iterate over the number of batches
        for m in range(0, numBatches):

            currBatchX = xData[m * batchSize: m * batchSize + batchSize]  # Extract batch X data
            currBatchY = yData[m * batchSize: m * batchSize + batchSize]  # Extract batch Y data

            # Train the neural net for a single step using the current batch's X and Y data
            train_step.run(feed_dict={x: currBatchX, y_: currBatchY, keep_prob: 0.5})

        # Calculate training accuracy after each epoch
        train_accuracy = accuracy.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0})  # Evaluate training accuracy
        print("Epoch %d, training accuracy %g" % (i, train_accuracy))
        trainAccuracies[i] = train_accuracy

        # Calculate testing accuracy after each epoch
        test_accuracy = accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0})  # Evaluate testing accuracy
        print("Epoch %d, testing accuracy %g" % (i, test_accuracy))
        testAccuracies[i] = test_accuracy


    sess.close()

    plt.plot(trainAccuracies, 'bo')  # Plot training accuracies as blue circles
    plt.plot(trainAccuracies, 'b--')  # Plot training accuracies as blue dashed lines
    plt.plot(testAccuracies, 'ro')  # Plot testing accuracies as red circles
    plt.plot(testAccuracies, 'r--')  # Plot training accuracies as red dashed lines
    plt.ylabel('Accuracy')  # Accuracy label
    plt.xlabel('Epoch')  # Epoch label

    #Lines for the legends
    train_line = mlines.Line2D([], [], linestyle='--', color='blue', marker='o', markersize=5, label='Training Data')
    test_line = mlines.Line2D([], [], linestyle='--', color='red', marker='o', markersize=5, label='Testing Data')

    # Draw legend at the upper left corner with legend lines
    plt.legend(handles=[train_line, test_line], loc='upper left', bbox_to_anchor=(0, 1))
    plt.title('Plot of Testing and Training Accuracies | Convolutional Neural Net')
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    midXLoc = (xmax - xmin)/2
    topYLoc = ymax - (ymax-ymin)*0.025
    plt.text(midXLoc, topYLoc, 'Num Epochs: ' + str(nEpochs) + '\nBatch Size: ' + str(batchSize), ha='center', va='top')
    #plt.show()  # Show plot
    plt.savefig('myfigConvNet', dpi=300)



# Initializes weight variables
def init_weights(shape):
  weights = tf.truncated_normal(shape, stddev=0.1)  # Assigns random weights from a truncated normal distribution.
  return tf.Variable(weights)

# Initializes bias variables
def init_biases(shape):
  biases = tf.constant(0.1, shape=shape)  # Assigns random biases from a truncated normal distribution.
  return tf.Variable(biases)

# Implements 2D convolution.
def conv2d(x, W):
  # Uses a stride of one and are zero padded so that the output is the same size as the input.
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # Pooling over 2x2 blocks


main()