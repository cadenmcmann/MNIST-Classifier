"""
Filename: MnistClassify.py
Author: Caden McMann
Date: 05-25-2021

"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def main():
    """
    Neural Network trained on MNIST data set
    Test accuracy: ~98.1%
    """

    # load data
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

    # Validation data will be 10% of training data
    num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
    num_validation_samples = tf.cast(num_validation_samples, tf.int64)  # cast to int

    # get number of test samples
    num_test_samples = mnist_info.splits['test'].num_examples
    num_test_samples = tf.cast(num_test_samples, tf.int64)  # cast to int

    # scale/standardize data using map function and method scaleData
    scaled_train_and_valid_data = mnist_train.map(scaleData)
    scaled_test_data = mnist_test.map(scaleData)

    # Shuffle training and validation data
    buffer = 10000
    shuffled_train_and_valid_data = scaled_train_and_valid_data.shuffle(buffer)

    # split validation vs training data
    validation_data = shuffled_train_and_valid_data.take(num_validation_samples)
    train_data = shuffled_train_and_valid_data.skip(num_validation_samples)

    # Batch training data to batches of size 100
    batch_size = 100
    train_data = train_data.batch(batch_size)
    # validation and test data each are wholly contained in 1 batch
    # done so train, validation, and test data all have same shape
    validation_data = validation_data.batch(num_validation_samples)
    test_data = scaled_test_data.batch(num_test_samples)

    # validation data must have same shape and object properties as
    # train and test data: Iterable and 2-tuple format (as_supervised=True)
    validation_inputs, validation_targets = next(iter(validation_data))

    # Data preprocessing over. Next step is build model

    input_size = 784
    output_size = 10
    hidden_layer_size = 200

    """
    Model Description:
        1. Input layer: Flatten converts 28x28 image vector to 784x1 input vector
        2. Two hidden layers: Multiply inputs by weights, add bias, and 
           apply non-linear activation function
        3. Output layer: Apply softmax activation to produce probability distribution
           values for each digit 0-9
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    # optimizer: 'Adam' with adjusted learning rate
    # sparse_categorical_crossentropy: applies one-hot encoding to output
    opt = tf.keras.optimizers.Adam(learning_rate=.0015)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    num_epochs = 10
    # train the model
    model.fit(train_data, epochs=num_epochs, validation_data=(validation_inputs, validation_targets), verbose=2)

    # Test model on test data. Print loss and accuracy of test data
    test_loss, test_accuracy = model.evaluate(test_data)
    final_loss = round(float(test_loss), 3)
    final_accuracy = round(float(test_accuracy), 3) * 100
    print('\nTest loss: ' + str(final_loss))
    print('Test accuracy: ' + str(final_accuracy) + '%')

    """
    A quick note: Good ML practice was applied in that the model and its hyper-parameters
    were not changed after evaluating the model on the test data. If the model were to be changed 
    after evaluating on test data, this may result in over-fitting, as the model has 'seen' the 
    test data already and such test data would then serve as extended training data
    """


def scaleData(image, label):
    '''
    Helper function to standardize our data. All values in image vector range
    between 0 and 255. Dividing by 255 standardizes the image vector so values
    range between 0 and 1.
    :param image: Vector to be standardized
    :param label: Label associated with image vector
    :return: standardized image vector and its label
    '''
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label


if __name__ == '__main__':
    main()