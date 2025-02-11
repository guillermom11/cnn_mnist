import numpy as np
from activationlayer import ActivationLayer
from fclayer import FCLayer
from loss import mse_prime, mse
from network import Network
from sigmoid import sigmoid_derivative,sigmoid
from keras.datasets import mnist
from keras.utils import to_categorical

def vectorize_image(matrix):
    # Flatten the matrix into a 1D array
    vectorized = matrix.flatten()

    # Add the bias term (1) at the beginning of the vector
    vectorized = np.insert(vectorized, 0, 1)

    # Reshape the vector into a row vector
    vectorized = vectorized.reshape(1, -1)

    return vectorized





(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train)
#print(x_train.shape)


# Apply vectorize_image to each matrix
vectorized_list = [vectorize_image(x) for x in x_train]


# Convert the list to a NumPy array
vectorized_dataset = np.array(vectorized_list)
#print(vectorized_dataset)
vectorized_dataset= vectorized_dataset.astype('float64')
vectorized_dataset /= 255
y_train = to_categorical(y_train)

# Call my Network class
network = Network()
network.add(FCLayer(785, 10))  # input_shape=(1, 785)    ;   output_shape=(1, 10)
network.add(ActivationLayer(sigmoid, sigmoid_derivative))

network.use(mse, mse_prime)
network.fit(vectorized_dataset, y_train, epochs=8, learning_rate=0.1)



vectorized_test = [vectorize_image(x) for x in x_test]
vectorized_test = np.array(vectorized_test)
vectorized_test = vectorized_test.astype('float64')
#print(vectorized_test)
vectorized_test /= 255

y_test = to_categorical(y_test)

output = network.predict(vectorized_test)
#print("predicted values : ")
#print(output, end="\n")
#print("true values : ")
#print(y_test)

# Use the output probabilities to get the predicted class (index of the maximum probability)
predicted_classes = [np.argmax(arr) for arr in output]
#print(predicted_classes)

# Use the true values to get the actual class
actual_classes = np.argmax(y_test, axis=1)

correct = 0
for i in range(len(predicted_classes)):
    if predicted_classes[i] == actual_classes[i]:
        correct += 1

accuracy = correct /len(actual_classes)
print(f"Accuracy: {accuracy *100}%")

