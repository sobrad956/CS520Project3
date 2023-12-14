import numpy as np
import random
from numpy.random import default_rng
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot
import pickle
#import pickle5 as pickle
#import pandas as pd




class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, hidden_size7, hidden_size8,  output_size, learning_rate, num_epochs, batch_size, model_type, distances_array, drop_out, real_data):
        #np.random.seed(0)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.hidden_size5 = hidden_size5
        self.hidden_size6 = hidden_size6
        self.hidden_size7 = hidden_size7
        self.hidden_size8 = hidden_size8
        self.output_size = output_size
        self.lr = learning_rate  #Haven't done anything smarter than standard SGD here
        self.epochs = num_epochs
        self.batch_size = batch_size 
        self.model_type = model_type  #Have to define differences in gradients in backprop function
        self.train_losses = []
        self.train_batch_losses = []
        self.test_losses = []
        self.test_batch_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        self.test_losses_smooth = []
        self.train_losses_smooth = []

        self.test_acc_smooth = []
        self.train_acc_smooth = []
        
        self.distances = distances_array
        self.real = real_data
        self.drop_out = drop_out #Percent of neurons at each layer to keep activated

    # initialize weights and biases
    def initialize_parameters(self):
        self.W1 = np.random.randn(self.hidden_size1, self.input_size) * math.sqrt(1.0/self.input_size) #np.zeros(self.hidden_size, self.input_size) #np.random.randn(self.hidden_size, self.input_size) * 0.01 #Haven't done anything interesting in initialization
        self.b1 = np.zeros((self.hidden_size1, 1))
        self.W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * math.sqrt(1.0/self.hidden_size1) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.hidden_size2, 1))
        self.W3 = np.random.randn(self.hidden_size3, self.hidden_size2) * math.sqrt(1.0/self.hidden_size2) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b3 = np.zeros((self.hidden_size3, 1))
        self.W4 = np.random.randn(self.hidden_size4, self.hidden_size3) * math.sqrt(1.0/self.hidden_size3) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b4 = np.zeros((self.hidden_size4, 1))
        self.W5 = np.random.randn(self.hidden_size5, self.hidden_size4) * math.sqrt(1.0/self.hidden_size4) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b5 = np.zeros((self.hidden_size5, 1))
        self.W6 = np.random.randn(self.hidden_size6, self.hidden_size5) * math.sqrt(1.0/self.hidden_size5) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b6 = np.zeros((self.hidden_size6, 1))
        self.W7 = np.random.randn(self.hidden_size7, self.hidden_size6) * math.sqrt(1.0/self.hidden_size6) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b7 = np.zeros((self.hidden_size7, 1))
        self.W8 = np.random.randn(self.output_size, self.hidden_size7) * math.sqrt(1.0/self.hidden_size7) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b8 = np.zeros((self.output_size, 1))
    
    #Activation & Loss Functions
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))
    
    def d_sigmoid(self, x):
    
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def tanh(self, x):
        return (2/(1+np.exp(-2*x))) - 1
    
    def d_tanh(self, x):
        return 1-(self.tanh(x)**2)
    
    def relu(self, x):
        return x * (x > 0)

    def d_relu(self, x):
        return 1.0 * (x > 0)
    
    def softmax(self, x): #For multiclass classification
        #return(np.exp(x)/np.exp(x).sum())
        return(np.exp(x)/(np.sum(np.exp(x), axis = 0)))
    
    def binary_cross_entropy_loss(self, y_pred, y_true): #For binary classification
        m = y_true.shape[0]
        return -(1/m) * np.sum(np.multiply(y_true, np.log(y_pred+ 1e-15) + (1-y_true)*np.log(1-y_pred+ 1e-15) ))
    
    def cross_entropy_loss(self, y_pred, y_true): #For multiclass classification
        m = y_true.shape[0]
        #print('y_true:', y_true.shape)
        #print('y_pred:', y_pred.shape)
        return -(1/m) * np.sum(np.multiply(y_true, np.log(y_pred +1e-15))) #Added small error for divide by zero errors

    # forward propagation
    def forward_propagation(self, X, test = False):    
        # compute the activation of the hidden layer
        #print(np.min(X))
        #print("before layer 1")
        #print(np.min(self.W1))
        Z1 = np.matmul(self.W1, X.T) + self.b1
        A1 = self.relu(Z1)

        #dropout
        if not test:
            A1shape = A1.shape[0]*A1.shape[1]
            self.D1 = np.ones(A1shape)
            self.D1[0:math.floor(self.drop_out * A1shape)] = 1
            np.random.shuffle(self.D1)
            self.D1 = self.D1.reshape((A1.shape[0], A1.shape[1]))
        A1 = np.multiply(self.D1, A1)

        #self.A1 = self.relu(self.Z1)
    
        # compute the activation of the output layer
        #print("before layer 2")
        Z2 = np.matmul(self.W2, A1) + self.b2
        #if self.model_type == 1:
        #   A2 = self.softmax(Z2)
        #else:
        A2 = self.relu(Z2)
        #dropout
        if not test:
            A2shape = A2.shape[0]*A2.shape[1]
            self.D2 = np.ones(A2shape)
            self.D2[0:math.floor(self.drop_out * A2shape)] = 1
            np.random.shuffle(self.D2)
            self.D2 = self.D2.reshape((A2.shape[0], A2.shape[1]))
        A2 = np.multiply(self.D2, A2)

        Z3 = np.matmul(self.W3, A2) + self.b3
        A3 = self.relu(Z3)
        #dropout
        if not test:
            A3shape = A3.shape[0]*A3.shape[1]
            self.D3 = np.ones(A3shape)
            self.D3[0:math.floor(self.drop_out * A3shape)] = 1
            np.random.shuffle(self.D3)
            self.D3 = self.D3.reshape((A3.shape[0], A3.shape[1]))
        A3 = np.multiply(self.D3, A3)

        Z4 = np.matmul(self.W4, A3) + self.b4
        A4 = self.relu(Z4)
        #dropout
        if not test:
            A4shape = A4.shape[0]*A4.shape[1]
            self.D4 = np.ones(A4shape)
            self.D4[0:math.floor(self.drop_out * A4shape)] = 1
            np.random.shuffle(self.D4)
            self.D4 = self.D4.reshape((A4.shape[0], A4.shape[1]))
        A4 = np.multiply(self.D4, A4)

        Z5 = np.matmul(self.W5, A4) + self.b5
        A5 = self.relu(Z5)
        #dropout
        if not test:
            A5shape = A5.shape[0]*A5.shape[1]
            self.D5 = np.ones(A5shape)
            self.D5[0:math.floor(self.drop_out * A5shape)] = 1
            np.random.shuffle(self.D5)
            self.D5 = self.D5.reshape((A5.shape[0], A5.shape[1]))
        A5 = np.multiply(self.D5, A5)

        Z6 = np.matmul(self.W6, A5) + self.b6
        A6 = self.relu(Z6)
        #dropout
        if not test:
            A6shape = A6.shape[0]*A6.shape[1]
            self.D6 = np.ones(A6shape)
            self.D6[0:math.floor(self.drop_out * A6shape)] = 1
            np.random.shuffle(self.D6)
            self.D6 = self.D6.reshape((A6.shape[0], A6.shape[1]))
        A6 = np.multiply(self.D6, A6)

        Z7 = np.matmul(self.W7, A6) + self.b7
        A7 = self.relu(Z7)
        #dropout
        if not test:
            A7shape = A7.shape[0]*A7.shape[1]
            self.D7 = np.ones(A7shape)
            self.D7[0:math.floor(self.drop_out * A7shape)] = 1
            np.random.shuffle(self.D7)
            self.D7 = self.D7.reshape((A7.shape[0], A7.shape[1]))
        A7 = np.multiply(self.D7, A7)

        Z8 = np.matmul(self.W8, A7) + self.b8
    
        
        #A3 = self.sigmoid(Z3)
        if self.model_type == 2:
            A8 = self.sigmoid(Z8)
        else:
            A8 = self.softmax(Z8)


        if test == False:
            self.Z1 = Z1
            self.A1 = A1
            self.Z2 = Z2
            self.A2 = A2
            self.Z3 = Z3
            self.A3 = A3

            self.Z4 = Z4
            self.A4 = A4
            self.Z5 = Z5
            self.A5 = A5
            self.Z6 = Z6
            self.A6 = A6
            self.Z7 = Z7
            self.A7 = A7
            self.Z8 = Z8
            self.A8 = A8
        
        return A8

    
    # backward propagation
    def backward_propagation(self, X, y):
        #pass
        m = y.shape[0]

        # compute the derivative of the loss with respect to A2 (output)
        #y = y.T
        
        #print('y shape: ', y.shape)
        #print('X shape: ', X.shape)
        #print('ypred shape: ', self.A2.shape)

        #loss = self.binary_cross_entropy_loss(self.A2, y)

        #print('Loss: ', loss)

        #print(self.A2.shape)
       
        if self.model_type == 1:
            dA8 = ((self.A8) - y)
        else:
            #dA3 = (self.A3 * (1-self.A3))
            dA8 = - (y/(self.A8)+1e-15) + ((1-y)/((1-self.A8)+1e-15))
        #if self.model_type == 1:
        #    dA3 = (self.softmax(self.A3) - y) #sus
        #else:
        #    dA3 = - (y/(self.A3)+1e-15) + ((1-y)/((1-self.A3)+1e-15)) #This is correct

        #print('dA2 shape: ', dA2.shape)
    
        # compute the derivative of the activation function of the output layer
        #dZ2 = dA2 * self.d_relu(self.Z2)
        dZ8 = dA8 * (self.A8 * (1-self.A8))
        self.dW8 = (1/m) * np.matmul(dZ8, self.A7.T)
        self.db8 = (1/m) * np.sum(dZ8, axis=1, keepdims=True)

        dA7 = np.dot(self.W8.T, dZ8)
        dZ7 = dA7 * self.d_relu(self.A7)
        dZ7 = np.multiply(dZ7, self.D7) #DROPOUT
        self.dW7 = (1/m) * np.matmul(dZ7, self.A6.T)
        self.db7 = (1/m) * np.sum(dZ7, axis=1, keepdims=True)
        
        dA6 = np.dot(self.W7.T, dZ7)
        dZ6 = dA6 * self.d_relu(self.A6)
        dZ6 = np.multiply(dZ6, self.D6) #DROPOUT
        self.dW6 = (1/m) * np.matmul(dZ6, self.A5.T)
        self.db6 = (1/m) * np.sum(dZ6, axis=1, keepdims=True)

        dA5 = np.dot(self.W6.T, dZ6)
        dZ5 = dA5 * self.d_relu(self.A5)
        dZ5 = np.multiply(dZ5, self.D5) #DROPOUT
        self.dW5 = (1/m) * np.matmul(dZ5, self.A4.T)
        self.db5 = (1/m) * np.sum(dZ5, axis=1, keepdims=True)

        dA4 = np.dot(self.W5.T, dZ5)
        dZ4 = dA4 * self.d_relu(self.A4)
        dZ4 = np.multiply(dZ4, self.D4) #DROPOUT
        self.dW4 = (1/m) * np.matmul(dZ4, self.A3.T)
        self.db4 = (1/m) * np.sum(dZ4, axis=1, keepdims=True)

        dA3 = np.dot(self.W4.T, dZ4)
        dZ3 = dA3 * self.d_relu(self.A3)
        dZ3 = np.multiply(dZ3, self.D3) #DROPOUT
        self.dW3 = (1/m) * np.matmul(dZ3, self.A2.T)
        self.db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(self.W3.T, dZ3)
        dZ2 = dA2 * self.d_relu(self.A2) #(self.A2 * (1-self.A2)) #This should be correct
        dZ2 = np.multiply(dZ2, self.D2) #DROPOUT
        # compute the derivative of the weights and biases of the output layer
        self.dW2 = (1/m) * np.matmul(dZ2, self.A1.T)
        self.db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
 
    
        # # compute the derivative of the activation function of the hidden layer
        dA1 = np.dot(self.W2.T, dZ2)
        #dZ1 = dA1 * self.d_sigmoid(self.Z1)
        #dZ1 = dA1 * self.d_relu(self.Z1)

        
        dZ1 = dA1 * self.d_relu(self.A1)
        dZ1 = np.multiply(dZ1, self.D1) #DROPOUT
            
            #(self.A1 * (1-self.A1))
        
        # print("dz1")
        # print(np.min(dZ1))
        # print()
        # # compute the derivative of the weights and biases of the hidden layer
        self.dW1 = (1/m) * np.dot(dZ1, X)
        self.db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    def update_parameters(self):
        # update the weights and biases
        self.W1 = self.W1 - self.lr * self.dW1
        self.b1 = self.b1 - self.lr * self.db1
        self.W2 = self.W2 - self.lr * self.dW2
        self.b2 = self.b2 - self.lr * self.db2
        self.W3 = self.W3 - self.lr * self.dW3
        self.b3 = self.b3 - self.lr * self.b3
        self.W4 = self.W4 - self.lr * self.dW4
        self.b4 = self.b4 - self.lr * self.b4
        self.W5 = self.W5 - self.lr * self.dW5
        self.b5 = self.b5 - self.lr * self.b5
        self.W6 = self.W6 - self.lr * self.dW6
        self.b6 = self.b6 - self.lr * self.b6
        self.W7 = self.W7 - self.lr * self.dW7
        self.b7 = self.b7 - self.lr * self.b7
        self.W8 = self.W8 - self.lr * self.dW8
        self.b8 = self.b8 - self.lr * self.b8
        
    
    def zero_grad(self):
        self.dW1 = 0
        self.dW2 = 0
        self.dW3 = 0
        self.dW4 = 0
        self.dW5 = 0
        self.dW6 = 0
        self.dW7 = 0
        self.dW8 = 0
        self.db1 = 0 
        self.db2 = 0
        self.db3 = 0
        self.db4 = 0
        self.db5 = 0
        self.db6 = 0
        self.db7 = 0
        self.db8 = 0
        
    
    # train the neural network


    def acc_score(self, y_actual, y_pred):
        if self.model_type == 1:
            #y_pred = self.softmax(y_pred)
            res = np.argmax(y_pred, axis=0)
            acc = 0
            for i in range(len(res)):
                if y_actual[res[i],i] == 1:
                    acc += 1.0
            acc = acc/y_pred.shape[1]
        else:
            res = y_pred > 0.5
            acc = 0
            #print(len(res))
            for i in range(res.shape[1]):
                #print("y_act", y_actual.shape)
                #print("res", res.shape)
                #print(y_actual[i] == res.T[i])
                if y_actual.T[i] == res.T[i]:
                    acc += 1.0
            acc = acc/y_pred.shape[1]
            #print(acc)
        return acc

    def train(self, X, y, X_test, y_test):
        # initialize the weights and biases
        self.initialize_parameters()
        
        distances = self.distances
    
        for i in range(self.epochs):
        #for i in range(1):
            batch_indices = random.sample([i for i in range(X.shape[0])], k = self.batch_size)
            test_batch_indices = random.sample([i for i in range(X_test.shape[0])], k = self.batch_size)
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]

            x_test_batch = X_test[test_batch_indices]
            y_test_batch = y_test[test_batch_indices]
            #print('actual', y_test_batch[0:2,:])
            
            if self.real:
                for elem in x_batch:
                    print(elem.shape)
                    print(elem[0])
                    idx = int(elem[0]*900)
                    elem = np.multiply(elem, distances[idx])
                    #pass
                    
                for elem in x_test_batch:
                    idx = int(elem[0]*900)
                    elem = np.multiply(elem, distances[idx])
                    #pass
                

            #x_batch = X
            #y_batch = y

            # forward propagation
            #print("before forward prob")
            self.forward_propagation(x_batch)

            
            y_batch = y_batch.T
            y_test_batch = y_test_batch.T
        
            # compute the loss
            #print("before loss compute")

            

            

            acc_tr = self.acc_score(y_batch, self.A8)
            loss = self.cross_entropy_loss(self.A8, y_batch)
            # if self.model_type == 1:
            #     loss = self.cross_entropy_loss(self.A2, y_batch)
            #     #loss = self.cross_entropy_loss(self.predict(x_batch), y_batch)
            # else:
            #     #loss = self.binary_cross_entropy_loss(self.A2, y_batch)
            #     #loss = self.binary_cross_entropy_loss(self.predict(x_batch), y_batch)
        
            # backward propagation
            #print("before back prob")
            self.zero_grad()
            self.backward_propagation(x_batch, y_batch)
        
            # update the parameters
            #print("before update param")
            self.update_parameters()
            
            #print(x_test_batch[0:2,:].shape)
            #print('prediction', self.predict(x_test_batch[0:2,:]))
            
            
            #print(f"iteration {i}: train loss = {loss}")
            #print("before predict")
            acc_ts = self.acc_score(y_test_batch, self.predict(x_test_batch))

            test_loss = self.cross_entropy_loss(self.forward_propagation(x_test_batch, True), y_test_batch)
            

            self.test_losses_smooth.append(test_loss)
            self.train_losses_smooth.append(loss)

            self.test_acc_smooth.append(acc_ts)
            self.train_acc_smooth.append(acc_tr)

            if i % 10 == 0:
                
                # self.predict(x_test_batch)
                #test_loss = self.binary_cross_entropy_loss(self.predict(X_test), y_test.T)
                #print(f"iteration {i}: Total train loss = {loss}")
                loss_smooth = np.mean(np.asarray(self.train_losses_smooth))
                test_loss_smooth = np.mean(np.asarray(self.test_losses_smooth))

                self.test_losses.append(test_loss_smooth)
                self.train_losses.append(loss_smooth)
                self.test_accuracies.append(np.mean(self.test_acc_smooth))
                self.train_accuracies.append(np.mean(self.train_acc_smooth))

                print(f"iteration {i}: Total train loss = {loss_smooth}, total test loss = {test_loss_smooth}")
                # print(f"iteration {i}: train loss = {loss}, test loss = {test_loss}")
                self.test_losses_smooth = []
                self.train_losses_smooth = []
                self.test_acc_smooth = []
                self.train_acc_smooth = []
                
                

            #print(self.A2)
            #print(self.b1)

    
    # predict the labels for new data
    def predict(self, X):
        
        #distances = self.distances
        #idx = int(X[0]*900)
        #if self.real:
        #    X = np.multiply(X, distances[idx])
        
        
        if self.model_type == 1:
            A3 = self.forward_propagation(X, True)
            pred = np.zeros((X.shape[0], 5))
            res = np.asarray([np.argmax(A3[:,i]) for i in range(A3.shape[1])])
            for i in range(len(res)):
                pred[i, res[i]] =1
            return pred.T
        else:
            A3 = self.forward_propagation(X, True)
            predictions = (A3 > 0.5).astype(int)
            return predictions

    def plotLoss(self):
        plt.plot(self.train_losses, label='train')
        plt.plot(self.test_losses, label='test')
        plt.legend(loc='best')
        plt.title('Loss Plot')
        plt.savefig('loss_plot.png')
        plt.show()
        plt.close()

    def plotAcc(self):
        plt.plot(self.test_accuracies, label='test')
        plt.plot(self.train_accuracies, label='train')
        plt.legend(loc='best')
        plt.title('Accuracy Plot')
        plt.savefig('acc.png')
        plt.show()
        plt.close()

    
    


def clean_data(dataset, train_split, model_type):

    boundary = int(math.ceil(dataset.shape[0]*train_split))

    rng = default_rng()
    idx = rng.choice(dataset.shape[0], size = boundary, replace=False)
    mask = np.ones(dataset.shape[0], dtype=bool)
    mask[idx] = False

    train = dataset[~mask, :]
    test = dataset[mask, :]

    
    #train = dataset[:boundary]
    #test = dataset[boundary:]

    X_train = train[:,0:3]
    if model_type == 1:
        y_train = train[:,3:4] #For model 1, predicting the move
    else:
        y_train = train[:,4:5] #For model 2, predict success

    X_test = test[:,0:3]
    if model_type == 1:
        y_test = test[:,3:4] #For model 1, predicting the move
    else:
        y_test = test[:,4:5]

    # print("Train shapes:")
    # print(X_train.shape)
    # print(y_train.shape)

    # print("Test shapes:")
    # print(X_test.shape)
    # print(y_test.shape)
    # print()

    print("Train Reshaped:")
    HH = np.hstack((np.concatenate(X_train[:,1]), np.concatenate(X_train[:,2]))).reshape(X_train.shape[0], 1256) #1260) #1252)
    X_train = np.hstack((X_train[:,0].reshape(-1, 1), HH))
    print(X_train.shape)
    if model_type == 1:
        y_train = np.concatenate(y_train[:,0]).reshape(X_train.shape[0], 5)
    else:
        #print(y_train)
        y_train = y_train[:,0].reshape(X_train.shape[0], 1)
    print(y_train.shape)

    print()

    print("Test Reshaped:")
    HH = np.hstack((np.concatenate(X_test[:,1]), np.concatenate(X_test[:,2]))).reshape(X_test.shape[0], 1256) #1260) #1252)
    X_test = np.hstack((X_test[:,0].reshape(-1, 1), HH))
    print(X_test.shape)
    if model_type == 1:
        y_test = np.concatenate(y_test[:,0]).reshape(X_test.shape[0], 5)
    else: 
        y_test = y_test[:,0].reshape(X_test.shape[0], 1)
    print(y_test.shape)
    return(X_train, y_train, X_test, y_test)


def model1(distances_array,train_split,   real_data,):
    model_type = 1
    dataset = np.load('dataframe.npy', allow_pickle=True)
    #print(dataset[0])
    # dataset = np.load('dataframe.npy', allow_pickle=True)
    # dataset = pd.DataFrame(dataset)
    # dataset = dataset.fillna(0.0)
    # dataset = dataset.to_numpy()
    
    if real_data:
        
        X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)
        #print(X_train[:,0])

        # with open('X_train.pickle', "wb") as b_file:
        #     pickle.dump(X_train, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('X_test.pickle', "wb") as b_file:
        #     pickle.dump(X_test, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('y_train.pickle', "wb") as b_file:
        #     pickle.dump(y_train, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('y_test.pickle', "wb") as b_file:
        #     pickle.dump(y_test, b_file, pickle.HIGHEST_PROTOCOL)


        # X_train[:,0] = X_train[:,0] / (30*30)
        # X_test[:,0] = X_test[:,0] / (30*30)
        # in_size = (len(dataset[0,1])*2)+1


        # temp = X_train.shape[0]
        # X = np.vstack((X_train,X_test))
        # X = X.astype(float)
        # print("here")
        # X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        # X = X.astype(float)
        # print('hi')

        # covariance_matrix = np.cov(X, ddof = 1, rowvar = False)
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # order_of_importance = np.argsort(eigenvalues)[::-1]
        # sorted_eigenvalues = eigenvalues[order_of_importance]
        # sorted_eigenvectors = eigenvectors[:,order_of_importance]

        # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        # k = 30 # select the number of principal components
        # reduced_data = np.matmul(X, sorted_eigenvectors[:,:k])
        # total_explained_variance = sum(explained_variance[:k])
        # print(total_explained_variance)

        # X_train = reduced_data[:temp,]
        # X_test = reduced_data[temp:,]

        # print(X_train.shape)
        # print(X_test.shape)

        # in_size = k
        drop_out = 0.7
        nn = NN(1261, 30, 25, 20, 18, 15, 12, 8, 6, 5, .1*math.sqrt(67), 2000, 67, model_type, distances_array, drop_out, True) #53, model_type)
    else:
        X_train = np.random.randn(59042, 1) # matrix of random x data
        y_train = np.zeros((59042, 5))
        for i in range(59042):
            if X_train[i] > 0.5:
                y_train[i,0] = 1 
            else:
                y_train[i,1] = 1 
        #y_train = X_train[:,1] > 0.5

        X_test = np.random.randn( 25303, 1) # matrix of random x data
        
        y_test = np.zeros((25303, 5))
        for i in range(25303):
            if X_test[i] > 0.5:
                y_test[i,0] = 1 
            else:
                y_test[i,1] = 1
        
        #nn = NN(1, 20, 10, 5, .1 ,1000, 53, model_type)
        drop_out = 0.7
        nn = NN(1, 10, 9, 8, 7, 6, 5, 4, 3, 5, .01*math.sqrt(53), 1000, 53, model_type, distances_array, drop_out, False)

    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

def model2(distances_array,train_split,  real_data):
    model_type = 2
    dataset = np.load('dataframe.npy', allow_pickle=True)
    
    if real_data:
        X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)


        # with open('X_train.pickle', "wb") as b_file:
        #     pickle.dump(X_train, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('X_test.pickle', "wb") as b_file:
        #     pickle.dump(X_test, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('y_train.pickle', "wb") as b_file:
        #     pickle.dump(y_train, b_file, pickle.HIGHEST_PROTOCOL)

        # with open('y_test.pickle', "wb") as b_file:
        #     pickle.dump(y_test, b_file, pickle.HIGHEST_PROTOCOL)




        X_train[:,0] = X_train[:,0] / (30*30)
        X_test[:,0] = X_test[:,0] / (30*30)
        in_size = (len(dataset[0,1])*2)+1

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)


        # temp = X_train.shape[0]
        # X = np.vstack((X_train,X_test))
        # X = X.astype(float)
        # print("here")
        # X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        # X = X.astype(float)
        # print('hi')

        # covariance_matrix = np.cov(X, ddof = 1, rowvar = False)
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # order_of_importance = np.argsort(eigenvalues)[::-1]
        # sorted_eigenvalues = eigenvalues[order_of_importance]
        # sorted_eigenvectors = eigenvectors[:,order_of_importance]

        # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        # k = 50 # select the number of principal components
        # reduced_data = np.matmul(X, sorted_eigenvectors[:,:k])
        # total_explained_variance = sum(explained_variance[:k])
        # print(total_explained_variance)

        # X_train = reduced_data[:temp,]
        # X_test = reduced_data[temp:,]

        # print(X_train.shape)
        # print(X_test.shape)

        # in_size = 5
        drop_out = 0.7
        nn = NN(in_size, 10, 9, 8, 7, 6, 5, 4, 3, 1, .01*math.sqrt(53), 1000, 53, model_type, distances_array, drop_out, True)
    else:
        X_train = np.random.randn(59042, 5) # matrix of random x data
        y_train = X_train[:,1] > 0.5

        X_test = np.random.randn( 25303, 5) # matrix of random x data
        y_test = X_test[:,1] > 0.5
        y_test = y_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        #print(y_test.shape)
        #print(y_train.shape)


        # temp = X_train.shape[0]
        # X = np.vstack((X_train,X_test))
        # X = X.astype(float)
        # print("here")
        # X = (X - X.mean(axis = 0)) / X.std(axis = 0)
        # X = X.astype(float)
        # print('hi')

        # covariance_matrix = np.cov(X, ddof = 1, rowvar = False)
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # order_of_importance = np.argsort(eigenvalues)[::-1]
        # sorted_eigenvalues = eigenvalues[order_of_importance]
        # sorted_eigenvectors = eigenvectors[:,order_of_importance]

        # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        # k = 2 # select the number of principal components
        # reduced_data = np.matmul(X, sorted_eigenvectors[:,:k])
        # total_explained_variance = sum(explained_variance[:k])
        # print(total_explained_variance)

        # X_train = reduced_data[:temp,]
        # X_test = reduced_data[temp:,]

        # print(X_train.shape)
        # print(X_test.shape)

        in_size = 5

        drop_out = 0.7
        nn = NN(in_size, 600, 500, 400, 300, 200, 100, 50, 25, 1, .01, 1000, 53, model_type, distances_array, drop_out, False)
        
        
    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

def model3(distances_array,train_split,  real_data):
    return 0
    model_type = 3
    dataset = np.load('dataframe.npy', allow_pickle=True)
    # Simulate initial board states
    X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)
    X_train[:,0] = X_train[:,0] / (30*30)
    X_test[:,0] = X_test[:,0] / (30*30)
    in_size = (len(dataset[0,1])*2)+1

    #y_train = np.squeeze(y_train)
    #y_test = np.squeeze(y_test)
    
    # Train model 1 on each board state to get predicted next move based on true success labels
    nn_actor = NN(1261, 30, 25, 20, 18, 15, 12, 8, 6, 5, .1*math.sqrt(67), 2000, 67, model_type, distances_array, True) #53, model_type)
    moves = nn_actor.predict(X)

    # Move all the states
    X = X.transition(moves)
    # Train model 2 on each board state to predict probability of success
    nn_critic = NN(in_size, 600, 100, 1, .1*math.sqrt(53), 1000, 53, model_type, distances_array, True)
    y = nn_critic.predict(X)

    # Repeat:
    for epoch in num_epochs:
        # Update model 1 using labels from model 2 instead of actual success labels
        nn_actor.train()
        moves = nn_actor.predict(X)
        # Move all states
        X = X.transition(moves)
        # Update model 2 on each board state to predict probability of success again
        nn_critic.train()
        y = nn_critic.predict(X)
    
    # Run a bot with model 1.
    bot3 = Bot(nn_actor)


def simulateData(k,boards):
    """This runs the 1 alien, 1 crew member experiments"""
    #numBoards = len(boards)
    numTrials = 100
    #bots = [1]
    success_flag = False
    
    shp = boards[0]
    alpha = 0.5
    data = []
    labels = []

    for trial in range(numTrials):
       
        i, j = shp.get_unoccupied_cell()
        bot = Bot(i, j, k, shp, 1, alpha)
        shp.bot = bot

        start_cells = []
        i, j = shp.get_unoccupied_cell()
        shp.ship[i][j].add_crew()
        start_cells.append(shp.ship[i][j])

        i, j = shp.get_unoccupied_alien_cell(k)
        alien = Alien(i, j, shp)

        #shp.print_ship()
        shp.distances_from_crew()
        # print("Calculated distances")
        shp.init_crew_prob_one()
        np.save('init_crew_probs.npy', shp.get_crew_probs())
        # print("init crew prob: ")
        # print(shp.get_crew_probs())
        #print()
                    
        shp.init_alien_prob_one()
        # print("init alien prob: ")
        # print(shp.get_alien_probs())
        #print()

        #shp.print_ship()
        print('Trial:', trial)
        T = 0
        flag = True
        while flag:
            datarow = []
            #if T > 40:
             #   break
            datarow.append(shp.ship[bot.row, bot.col].one_d_idx)

            ap = shp.get_alien_probs()
            cp = shp.get_crew_probs()

            

            ap = ap[shp.open_cell_indices()].flatten()
            cp = cp[shp.open_cell_indices()].flatten()

            datarow.append(ap)
            datarow.append(cp)

            decision = bot.bot1_move()
            decision_vector = np.zeros(5)
            decision_vector[decision] = 1

            datarow.append(decision_vector)
            #print(datarow)
            datarow = np.asarray(datarow)
            data.append(datarow)

            #shp.print_ship()
            #print()
            #print(shp.ship[bot.row, bot.col].one_d_idx)
            #print()
            
            
            # MUST RECORD THE MOVE IT TAKES
            i = bot.row
            j = bot.col

            if shp.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                labels.append(np.zeros(T+1))
                flag = False
                break
            if shp.ship[i][j].contains_crew():
                print(f"Saved: {T}")

                labels.append(np.ones(T+1))

                shp.ship[i][j].remove_crew()
                flag = False
                break

            shp.one_one_bot_move_update()
            #print("bot move: ", shp.get_crew_probs())

            if alien.move():
                print(f"Dead: {T}")
                labels.append(np.zeros(T+1))
                flag = False
                break
            shp.one_one_alien_move_update()
                        
            alien_beep = bot.detect_alien()
            shp.one_one_alien_beep_update(alien_beep)
            crew_beep = bot.detect_crew(start_cells)
            shp.one_one_crew_beep_update(crew_beep)
            T += 1
        shp.empty_ship()
        #bot_loc, all alien states, all crew states, move list, 
    
    data = np.asarray(data)
    labels = np.concatenate(labels).ravel().T.reshape(-1,1)
    print(data.shape)
    print(labels.shape)
    dataframe = np.hstack((data, labels))
    #print(dataframe[-2:])
    np.save('dataframe.npy', dataframe)
    


def runSimulate():
    k = 3
    boards = []
    print("top of main")
    for i in range(1):
        #ship takes in k, D
        shp = Ship()
        shp.generate_ship()
        print("ship generated")
        boards.append(shp)
    #experiement takes k, boards
    with open('board.pickle', "wb") as b_file:
        pickle.dump(shp, b_file, pickle.HIGHEST_PROTOCOL)
    np.save('board.npy', boards[0])
    simulateData(k, boards)


def compareBots(model):
    """This runs bot 1 with and without neural network"""
    numTrials = 2
    k = 3
    prob_success = np.zeros((2, numTrials))
    avg_trial_len = np.zeros((2, numTrials))
    
    success_flag = False
    
    shp1 = Ship("board.npy")
    shp2 = Ship("board.npy")
    alpha = 0.5

    for trial in range(numTrials):
       
        i, j = shp1.get_unoccupied_cell()
        bot1 = Bot(i, j, k, shp1, 1, alpha)
        bot2 = Bot(i, j, k, shp2, 2, alpha)
        shp1.bot = bot1
        shp2.bot = bot2

        start_cells = []
        i, j = shp1.get_unoccupied_cell()
        shp1.ship[i][j].add_crew()
        shp2.ship[i][j].add_crew()
        start_cells.append(i)
        start_cells.append(j)

        i, j = shp1.get_unoccupied_alien_cell(k)
        alien1 = Alien(i, j, shp1)
        alien2 = Alien(i, j, shp2)

        shp1.distances_from_crew()
        shp2.distances_from_crew()
        # print("Calculated distances")
        shp1.init_crew_prob_one()
        shp2.init_crew_prob_one()
                    
        shp1.init_alien_prob_one()
        shp2.init_alien_prob_one()

        
        print('Trial:', trial)

        #Run bot1
        T = 0
        flag = True
        while flag:

            bot1.bot1_move()
            i = bot1.row
            j = bot1.col

            if shp1.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                avg_trial_len[0][trial] += T / (numTrials)
                flag = False
                break
            if shp1.ship[i][j].contains_crew():
                print(f"Saved: {T}")
                prob_success[0][trial] += 1 / numTrials
                avg_trial_len[0][trial] += T / (numTrials)
                shp1.ship[i][j].remove_crew()
                flag = False
                break
            shp1.one_one_bot_move_update()
            if alien1.move():
                print(f"Dead: {T}")
                avg_trial_len[0][trial] += T / (numTrials)
                flag = False
                break
            shp1.one_one_alien_move_update()
            alien_beep = bot1.detect_alien()
            shp1.one_one_alien_beep_update(alien_beep)
            crew_beep = bot1.new_detect_crew(start_cells[0], start_cells[1])
            shp1.one_one_crew_beep_update(crew_beep)
            T += 1
        shp1.empty_ship()


        #Run bot2
        T = 0
        flag = True
        while flag:
            
            # Neural Network powered bot move sequence
            cur_board_state = bot2.get_cur_state()
            pred = model.predict(cur_board_state)
            bot2.nn_bot_move(pred)
            
            # MUST RECORD THE MOVE IT TAKES
            i = bot2.row
            j = bot2.col

            if shp2.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                avg_trial_len[1][trial] += T / (numTrials)
                flag = False
                break
            if shp2.ship[i][j].contains_crew():
                print(f"Saved: {T}")
                prob_success[1][trial] += 1 / numTrials
                avg_trial_len[1][trial] += T / (numTrials)
                shp2.ship[i][j].remove_crew()
                flag = False
                break

            shp2.one_one_bot_move_update()
            #print("bot move: ", shp.get_crew_probs())

            if alien2.move():
                print(f"Dead: {T}")
                flag = False
                break
            shp2.one_one_alien_move_update()
                        
            alien_beep = bot2.detect_alien()
            shp2.one_one_alien_beep_update(alien_beep)
            crew_beep = bot2.new_detect_crew(start_cells[0], start_cells[1])
            shp2.one_one_crew_beep_update(crew_beep)
            T += 1
        shp2.empty_ship()
    

    
if __name__ == "__main__":
    #x = np.load('Final/board.npy', allow_pickle=True)
    # open a file, where you stored the pickled data

    file = open('board.pickle', 'rb')
    shp = pickle.load(file)
    
    file.close()
    shp.open_cell_indices()
    shp.open_cell_distances()
    #print(shp.print_ship())
    distances = shp.get_one_distances()
    print(distances.shape)
    

    # # dump information to that file
    #data = np.load("dataframe.npy", allow_pickle=True)

    # # close the file
    

    #dat = data[0:5000,]

    #np.save("dataframe_small.npy", dat)




    #with open('X_small.pickle', "wb") as b_file:
    #    pickle.dump(dat, b_file, pickle.HIGHEST_PROTOCOL)


    # print(data.print_ship())



    #runSimulate()
    #nn = model1(distances, train_split=0.7, real_data = True)
    nn = model1(distances,train_split=0.7,  real_data = False)
    
    #nn = model2(distances,train_split=0.7,  real_data = True)
    #nn = model2(distances,train_split=0.7,  real_data = False)
    #compareBots(nn)
    
    