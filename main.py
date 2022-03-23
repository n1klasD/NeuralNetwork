import numpy
from matplotlib import pyplot
import NeuralNetwork
import matplotlib.pyplot


def drawGreyImage(sizeX, sizeY, Pixels):
    image_array = numpy.asfarray(Pixels[1:]).reshape((sizeX, sizeY))
    matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
    pyplot.show()


def scale(data, maxValue):
    """Scales a list of corresponding data to the interval 0.01 - 1.0 """
    return (data / float(maxValue) * 0.99) + 0.01


if __name__ == "__main__":
    network = NeuralNetwork.neuralNetwork(784, 100, 10, 0.1)

    # training data
    data_file = open("mnist_dataset/mnist_train.csv", "r")
    training_data = []

    for i in range(0, 60000):
        training_data.append(data_file.readline().replace("\n", ""))
        print("Read progress:", i)
    data_file.close()

    n = 1

    for record in training_data:
        all_values = record.split(',')  # split data by commas
        # drawGreyImage(28, 28, all_values)  # draw current record
        scaled_input = scale(numpy.asfarray(all_values[1:]), 255.0)  # scale inputs --> range from 0.01 to 1
        # create target nodes for record, all 0.01, the correct number is 0.99
        targets = numpy.zeros(network.onodes) + 0.01
        targets[int(all_values[0])] = 0.99

        print("Training:", n, "/", len(training_data))
        n += 1

        # train network
        network.train(scaled_input, targets)
        pass

    # network = NeuralNetwork.neuralNetwork.load("network_serialised.bin")
    network.save()

    # test network
    network.test(10000)

    # scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale inputs --> range from 0.01 to 1
    # print(network.query(scaled_input))
