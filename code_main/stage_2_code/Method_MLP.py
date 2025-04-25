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

from code_main.stage_2_code.Evaluate_F1 import Evaluate_F1


class Method_MLP(method, nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 9e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        print("using device:", self.device)
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 128).to(self.device)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU().to(self.device)
        self.fc_layer_2 = nn.Linear(128, 128).to(self.device)
        self.activation_func_2 = nn.ReLU().to(self.device)
        self.fc_layer_3 = nn.Linear(128, 32).to(self.device)
        self.activation_func_3 = nn.ReLU().to(self.device)
        self.fc_layer_4 = nn.Linear(32, 10).to(self.device)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # Using LogSoftmax + NLLLoss is often more numerically stable than Softmax + CrossEntropyLoss
        self.activation_func_4 = nn.LogSoftmax(dim=1).to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        h = self.activation_func_3(self.fc_layer_3(h))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_4(self.fc_layer_4(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        self.to(self.device)
        training_metrics = []

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # loss_function = nn.CrossEntropyLoss()
        # Using NLLLoss with LogSoftmax output
        loss_function = nn.NLLLoss()

        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator accuracy', '')
        f1_evaluator = Evaluate_F1('training evaluator f1', '')
        precision_evaluator = Evaluate_F1('training evaluator precision', '')
        recall_evaluator = Evaluate_F1('training evaluator recall', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X).cpu()
            # convert y to torch.tensor as well
            y_true = y.cpu().long()
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

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            f1_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            precision_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            recall_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}

            accuracy = accuracy_evaluator.evaluate()
            f1_score = f1_evaluator.evaluate()
            precision = precision_evaluator.evaluate()
            recall = recall_evaluator.evaluate()
            loss = train_loss.item()

            training_metrics.append({
                "epoch": epoch,
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "loss": loss
            })

            if epoch % 40 == 0:
                print('Epoch:', epoch, 'Accuracy:', accuracy, 'F1:', f1_score,
                      'Precision:', precision,
                      'Recall:', recall, loss)

        return training_metrics

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device)).cpu()
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        training_metrics = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], "training_metrics": training_metrics}
