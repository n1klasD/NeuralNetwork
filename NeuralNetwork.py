import numpy
import scipy.special
import dill


class neuralNetwork:

    # initialize the network
    def __init__(self, inputnodes, hiddennotes, outputnodes, learningrate):
        # set network size
        self.filename = "network_serialised.bin"
        self.inodes = inputnodes
        self.hnodes = hiddennotes
        self.onodes = outputnodes
        self.scorecard = []

        # learning rate
        self.lr = learningrate

        # link weight matrices, w_ih and w_ho
        # weights inside the the arrays are w_i_j, where link is from node i to node j in the next layer
        # uses standard deviation with e = 0 and sd = 1/root(number of connections per per nodes of previous layer)

        self.w_ih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.w_ho = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # activation function --> sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # -- Part1: calculate output --

        # convert input to 2D Array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate input of first hidden layer
        hidden_inputs = numpy.dot(self.w_ih, inputs)

        # calculate the output of the first layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate input of final Layer
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)

        # signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)

        # -- Part2: calculate change based on Difference --
        # error is (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights and recombined
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        # update weights for links between the hidden and output layers
        self.w_ho += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                         numpy.transpose(hidden_outputs))

        # update weights for links between the input and hidden layers
        self.w_ih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                         numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate input of first hidden layer
        hidden_inputs = numpy.dot(self.w_ih, inputs)

        # calculate the output of the first layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate input of final Layer
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)

        # signals emerging from final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save(self):
        with open(self.filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return dill.load(f)

    def test(self, number):
        """tests the neural network with test_data and gives it a score"""
        # test data
        data_file = open("mnist_dataset/mnist_test.csv", "r")
        test_data = []

        for i in range(0, number):
            test_data.append(data_file.readline().replace("\n", ""))
        data_file.close()

        # reset scorecard
        self.scorecard = []

        for record in test_data:
            all_values = record.split(",")
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.query(inputs)
            label = numpy.argmax(outputs)  # index of the highest values corresponds to the label

            if label == correct_label:
                self.scorecard.append(1)
            else:
                self.scorecard.append(0)
                pass
            print("Testing: correct answer: " + str(correct_label) + " networks answer: " + str(label))
            pass
        print("Performance = ", numpy.asarray(self.scorecard).sum() / len(self.scorecard))
