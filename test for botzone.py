import json
import shelve
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))
        import shelve
        learning_results = shelve.open('../output')
        learning_results['weight_end'] = [i for i in self.weights]
        learning_results['bias_end'] = [i for i in self.biases]
        learning_results.close()


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
def procstep(grid,color_num,action):
    x0 = action['x0']
    y0 = action['y0']
    x1 = action['x1']
    y1 = action['y1']
    if x0<0:
        return 0
    grid[x1][y1] = color_num
    if abs(x0-x1)>1 or abs(y0-y1)>1:
        grid[x0][y0] = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if (i==0 and j==0) or x1+i<0 or x1+i>6 or y1+j<0 or y1+j>6:
                continue
            if grid[x1+i][y1+j] == -color_num:
                grid[x1+i][y1+j] = color_num

def action_pos():
    count = 0
    action_pos_dict = {}
    for i in range(7):
        for j in range(7):
            for a in range(-2,3):
                for b in range(-2,3):
                    if (a==0 and b==0) or i+a<0 or i+a>=7 or j+b<0 or j+b>=7:
                        continue
                    action = {}
                    action['x0'] = i+a
                    action['y0'] = j+b
                    action['x1'] = i
                    action['y1'] = j
                    action_pos_dict[count] = action
                    count += 1
    return action_pos_dict




string = json.loads(input())
responses = string['responses']
requests = string['requests']
if requests[0]['x0'] < 0:
    botColor = 1
    #netData = shelve.open('./data/blackData')
    netData = shelve.open('E:/blackData')
else:
    botColor = -1
    #netData = shelve.open('./data/whiteData')
    netData = shelve.open('E:/whiteData')

netWork = Network([49, 100, 200, 792])
netWork.weights = netData['weights']
netWork.biases = netData['biaes']
state = np.zeros((7,7))
state[0][0],state[6][6],state[0][6],state[6][0] = 1,1,-1,-1
for op1, op2 in zip(requests,responses):
    procstep(state,-botColor,op1)
    procstep(state,botColor,op2)
procstep(state,-botColor,requests.pop())
print(state)
inputArray = state.reshape((49,1)).copy()
outArray = netWork.feedforward(inputArray)
action_pos_dict = action_pos()
bestValue = -10000.0
ret = {}
for i in range(len(action_pos_dict)):
    action = action_pos_dict[i]
    x0 = action['x0']
    y0 = action['y0']
    x1 = action['x1']
    y1 = action['y1']
    if state[x0][y0] == botColor and state[x1][y1] == 0:
        value = outArray[i][0]
        if value > bestValue:
            bestValue = value
            ret['response'] = action
ret['debug'] = bestValue
out = json.dumps(ret)
print(out)







