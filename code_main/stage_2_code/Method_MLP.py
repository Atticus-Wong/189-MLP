'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_main.base_class.method import method
from code_main.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100  # Reduced epochs for potentially faster training, adjust as needed
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        input_size = 784
        hidden_size = 128
        output_size = 10

        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(input_size, hidden_size)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(hidden_size, output_size)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # Using LogSoftmax + NLLLoss is often more numerically stable than Softmax + CrossEntropyLoss
        self.activation_func_2 = nn.LogSoftmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # Input layer to hidden layer
        # Ensure input is float and normalized if necessary (assuming normalization happens in data loading or preprocessing)
        x = x.float() / 255.0  # Normalize pixel values to [0, 1]
        h = self.activation_func_1(self.fc_layer_1(x))
        # Hidden layer to output layer
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # Using NLLLoss with LogSoftmax output
        loss_function = nn.NLLLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            # Convert X to a NumPy array first if it's a list of lists, then to a FloatTensor
            X_tensor = torch.FloatTensor(np.array(X))
            y_pred = self.forward(X_tensor)
            # convert y to torch.tensor as well
            # Ensure y is a LongTensor for NLLLoss
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            # Evaluate and print every few epochs
            if epoch % 20 == 0 or epoch == self.max_epoch - 1:
                with torch.no_grad():  # No need to track gradients for evaluation
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                    accuracy = accuracy_evaluator.evaluate()
                    print(f'Epoch: {epoch:03d}, Loss: {train_loss.item():.4f}, Accuracy: {accuracy:.4f}')

    def test(self, X):
        # do the testing, and result the result
        # Convert X to a NumPy array first if it's a list of lists, then to a FloatTensor
        X_tensor = torch.FloatTensor(np.array(X))
        # Ensure the model is in evaluation mode (affects layers like dropout, batch norm)
        self.eval()
        with torch.no_grad():  # Disable gradient calculation for testing
            y_pred = self.forward(X_tensor)
        # Set the model back to train mode
        self.train()
        # convert the probability distributions (or log probabilities) to the corresponding labels
        # instances will get the labels corresponding to the largest probability/log probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        # Ensure data is loaded correctly before training
        if self.data is None or 'train' not in self.data or 'X' not in self.data['train'] or 'y' not in self.data['train']:
            raise ValueError("Training data not found or not loaded correctly.")
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        # Ensure test data is available
        if 'test' not in self.data or 'X' not in self.data['test'] or 'y' not in self.data['test']:
            raise ValueError("Test data not found or not loaded correctly.")
        pred_y = self.test(self.data['test']['X'])
        # Ensure true_y for test set is available for evaluation comparison if needed later
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
