from JSML.basic import *
import numpy as np
import os
import cv2

# Breakpoint

def load_mnist_dataset(dataset, path):
    # Create lists for samples and labels
    X, y = [], []

    # For each label folder
    for label in os.listdir(os.path.join(path, dataset)):
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(
                path, dataset, label, file), cv2.IMREAD_GRAYSCALE)
            # And append it and a label to the lists
            X.append(image)
            y.append(int(label))

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y)


def create_data_mnist(path):
    # Load both sets separately
    X_train, y_train = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    return X_train, y_train, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist(
    'resources/fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

model.evaluate(X_test, y_test)