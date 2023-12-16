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
import copy




class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, hidden_size7, hidden_size8,  output_size, learning_rate, decay, num_epochs, batch_size, model_type, distances_array, drop_out, real_data):
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
        self.decay = decay

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
        return (1 / (1 + np.exp(-x.astype(float))))
    
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
        return(np.exp(x.astype('float'))/(np.sum(np.exp(x.astype('float')), axis = 0)))
    
    def binary_cross_entropy_loss(self, y_pred, y_true): #For binary classification
        m = y_true.shape[1]
        return -(1/m) * np.sum(np.multiply(y_true, np.log(y_pred+ 1e-15) + (1-y_true)*np.log(1-y_pred+ 1e-15) ))
    
    def cross_entropy_loss(self, y_pred, y_true): #For multiclass classification
        m = y_true.shape[1]
        return -(1/m) * np.sum(np.multiply(y_true, np.log(y_pred +1e-15))) #Added small error for divide by zero errors

    # forward propagation
    def forward_propagation(self, X, test = False, deploy = False):    
        # compute the activation of the hidden layer

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

    
        # compute the activation of the output layer
        Z2 = np.matmul(self.W2, A1) + self.b2

        A2 = self.relu(Z2)
        #dropout
        if not test:
            A2shape = A2.shape[0]*A2.shape[1]
            self.D2 = np.ones(A2shape)
            self.D2[0:math.floor(self.drop_out * A2shape)] = 1
            np.random.shuffle(self.D2)
            self.D2 = self.D2.reshape((A2.shape[0], A2.shape[1]))
        #if not deploy:
            A2 = np.multiply(self.D2, A2)

        Z3 = np.matmul(self.W3, A2) + self.b3
        #print('Z3:',Z3.shape)
        A3 = self.relu(Z3)
        #dropout
        if not test:
            A3shape = A3.shape[0]*A3.shape[1]
            self.D3 = np.ones(A3shape)
            self.D3[0:math.floor(self.drop_out * A3shape)] = 1
            np.random.shuffle(self.D3)
            self.D3 = self.D3.reshape((A3.shape[0], A3.shape[1]))
        #if not deploy:
            A3 = np.multiply(self.D3, A3)

        Z4 = np.matmul(self.W4, A3) + self.b4
        #print('Z4:',Z4.shape)
        A4 = self.relu(Z4)
        #dropout
        if not test:
            A4shape = A4.shape[0]*A4.shape[1]
            self.D4 = np.ones(A4shape)
            self.D4[0:math.floor(self.drop_out * A4shape)] = 1
            np.random.shuffle(self.D4)
            self.D4 = self.D4.reshape((A4.shape[0], A4.shape[1]))
        #if not deploy:
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
        #if not deploy:
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
        #if not deploy:
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
        #if not deploy:
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


        if self.model_type == 1:
            dA8 = ((self.A8) - y)
        else:
            #dA3 = (self.A3 * (1-self.A3))
            dA8 = - (y/(self.A8)+1e-15) + ((1-y)/((1-self.A8)+1e-15))
        
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
            

        # # compute the derivative of the weights and biases of the hidden layer
        self.dW1 = (1/m) * np.dot(dZ1, X)
        self.db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    def update_parameters(self):
        # update the weights and biases
        lr = self.lr/(1+self.decay*(self.i+ (self.batches_num*self.e)))
        #print(lr)
        self.W1 = self.W1 - lr * self.dW1
        self.b1 = self.b1 - lr * self.db1
        self.W2 = self.W2 - lr * self.dW2
        self.b2 = self.b2 - lr * self.db2
        self.W3 = self.W3 - lr * self.dW3
        self.b3 = self.b3 - lr * self.b3
        self.W4 = self.W4 - lr * self.dW4
        self.b4 = self.b4 - lr * self.b4
        self.W5 = self.W5 - lr * self.dW5
        self.b5 = self.b5 - lr * self.b5
        self.W6 = self.W6 - lr * self.dW6
        self.b6 = self.b6 - lr * self.b6
        self.W7 = self.W7 - lr * self.dW7
        self.b7 = self.b7 - lr * self.b7
        self.W8 = self.W8 - lr * self.dW8
        self.b8 = self.b8 - lr * self.b8
        
    
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
        X_shuff = np.hstack((X, y))
        np.random.shuffle(X_shuff)
        if self.model_type == 1:
            X = X_shuff[:,:-5]
            y = X_shuff[:,-5:]
        else:
            X = X_shuff[:,:-1]
            y = X_shuff[:,-1:]

        X_shuff = np.hstack((X_test, y_test))
        np.random.shuffle(X_shuff)
        if self.model_type == 1:
            X_test = X_shuff[:,:-5]
            y_test = X_shuff[:,-5:]
        else:
            X_test = X_shuff[:,:-1]
            y_test = X_shuff[:,-1:]

        
        
        self.batches_num = math.floor(X.shape[0]/self.batch_size)
        print('num batches:', self.batches_num)
        test_batch_size = int(math.floor(X_test.shape[0]/self.batches_num))
        y_t = y.T
        for self.e in range(1,self.epochs+1):
            start = 0
            end = self.batch_size

            start_test = 0
            end_test = test_batch_size
            for self.i in range(1,int(self.batches_num)):
            #for i in range(1):
                #batch_indices = random.sample([i for i in range(X.shape[0])], k = self.batch_size)
                #test_batch_indices = random.sample([i for i in range(X_test.shape[0])], k = self.batch_size)
                x_batch = X[start:end,]
                y_batch = y[start:end,]

                x_test_batch = X_test[start_test:end_test,]
                y_test_batch = y_test[start_test:end_test,]
           

                #x_batch = X
                #y_batch = y

                # forward propagation
                
                # if self.real:
                #     for elem in x_batch:
                #         #print(elem.shape)
                #         #print(elem[0])
                #         idx = int(elem[0]*900)
                #         elem = np.multiply(elem, distances[idx])
                #         #pass
                        
                #     for elem in x_test_batch:
                #         idx = int(elem[0]*900)
                #         elem = np.multiply(elem, distances[idx])
                #         #pass
                
                self.forward_propagation(x_batch)

                
                y_batch = y_batch.T
                y_test_batch = y_test_batch.T
            
                # compute the loss
                #print("before loss compute")

                

                

                #acc_tr_batch = self.acc_score(y_batch, self.A8)
 
                #batch_loss = self.cross_entropy_loss(self.A8, y_batch)
                loss = self.cross_entropy_loss(self.forward_propagation(x_batch, True), y_batch)
                acc_tr = self.acc_score(y_batch, self.predict(x_batch))
         
            
                # backward propagation

                self.zero_grad()
                self.backward_propagation(x_batch, y_batch)
            
                # update the parameters

                self.update_parameters()
         
                acc_ts = self.acc_score(y_test_batch, self.predict(x_test_batch))

                test_loss = self.cross_entropy_loss(self.forward_propagation(x_test_batch, True), y_test_batch)
                

                self.test_losses_smooth.append(test_loss)
                self.train_losses_smooth.append(loss)

                self.test_acc_smooth.append(acc_ts)
                self.train_acc_smooth.append(acc_tr)

                if self.i % 10 == 0:
                    
                    # self.predict(x_test_batch)
                    #test_loss = self.binary_cross_entropy_loss(self.predict(X_test), y_test.T)
                    #print(f"iteration {i}: Total train loss = {loss}")
                    loss_smooth = np.mean(np.asarray(self.train_losses_smooth))
                    test_loss_smooth = np.mean(np.asarray(self.test_losses_smooth))

                    self.test_losses.append(test_loss_smooth)
                    self.train_losses.append(loss_smooth)
                    self.test_accuracies.append(np.mean(self.test_acc_smooth))
                    self.train_accuracies.append(np.mean(self.train_acc_smooth))

                    print(f"epoch {self.e} iteration {self.i}: Total train loss = {loss_smooth}, total test loss = {test_loss_smooth}")
                    # print(f"iteration {i}: train loss = {loss}, test loss = {test_loss}")
                    self.test_losses_smooth = []
                    self.train_losses_smooth = []
                    self.test_acc_smooth = []
                    self.train_acc_smooth = []
                    
                    

                start = end
                end += self.batch_size

                start_test = end_test
                end_test += test_batch_size
        self.i = 0
        self.e = 0

    
    # predict the labels for new data
    def predict(self, X, deploy = False):
        
        distances = self.distances

        
        # #idx = int(X[0])
        # if self.real:
        #    for elem in X:
        #         idx = int(elem[0]*900)
        #         elem = np.multiply(elem, distances[idx])
        #         #pass
        
        
        if self.model_type == 1:
            A3 = self.forward_propagation(X, True, deploy)
            pred = np.zeros((X.shape[0], 5))
            res = np.asarray([np.argmax(A3[:,i]) for i in range(A3.shape[1])])
            for i in range(len(res)):
                pred[i, res[i]] =1
            return pred.T
        else:
            A3 = self.forward_propagation(X, True, deploy)
            predictions = (A3 > 0.5).astype(int)
            return predictions

    def plotLoss(self):
        plt.plot(self.train_losses, label='train')
        plt.plot(self.test_losses, label='test')
        plt.legend(loc='best')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epoch Number (in 10s)')
        plt.ylabel('Cross-Entropy Loss')
        plt.savefig('loss_plot.png')
        plt.show()
        plt.close()

    def plotAcc(self):
        #plt.figure(figsize=(6.4,8))
        plt.plot(self.test_accuracies, label='test')
        plt.plot(self.train_accuracies, label='train')
        plt.legend(loc='best')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epoch Number (in 10s)')
        plt.ylabel('Percent Accuracy')
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

    # print("PRE-BALANCE")
    # print("Train shapes:")
    # print(X_train.shape)
    # print(y_train.shape)

    # print("Test shapes:")
    # print(X_test.shape)
    # print(y_test.shape)
    # print()

    # print("Train Reshaped:")
    #print("test", X_train[:,1].shape)
    #print(X_train[:,1].shape)
    print(X_train.shape)
    #np.concatenate(X_train[:,1])
    print(X_train[:,2])

    HH = np.hstack((np.concatenate(X_train[:,1]), np.concatenate(X_train[:,2]))).reshape(X_train.shape[0], 8)
    #HH = np.hstack(((np.expand_dims(X_train[:,1],0)),np.expand_dims(X_train[:,2],0))).reshape(X_train.shape[0], 1800)
    X_train = np.hstack((X_train[:,0].reshape(-1, 1), HH))
    #print(X_train.shape)
    if model_type == 1:
        y_train = np.concatenate(y_train[:,0]).reshape(X_train.shape[0], 5)
    else:
        #print(y_train)
        y_train = y_train[:,0].reshape(X_train.shape[0], 1)
    # print(y_train.shape)

    # print()

    # print("Test Reshaped:")
    HH = np.hstack((np.concatenate(X_test[:,1]), np.concatenate(X_test[:,2]))).reshape(X_test.shape[0], 8)

    X_test = np.hstack((X_test[:,0].reshape(-1, 1), HH))
    # print(X_test.shape)
    if model_type == 1:
        y_test = np.concatenate(y_test[:,0]).reshape(X_test.shape[0], 5)
    else: 
        y_test = y_test[:,0].reshape(X_test.shape[0], 1)
    # print(y_test.shape)
    
    X_train, y_train = balanceSet(X_train, y_train, model_type)
    X_test, y_test = balanceSet(X_test, y_test, model_type)

    print("Train shapes:")
    #print(X_train.shape)
    #print(y_train.shape)

    print("Test shapes:")
    #print(X_test.shape)
    #print(y_test.shape)
    print()
    return(X_train, y_train, X_test, y_test)


def model1(distances_array,train_split, real_data,):
    model_type = 1
    adj = np.load('Actual/data_adjacent.npy', allow_pickle=True)
    #dataset = np.load('Actual/dataframe.npy', allow_pickle=True)
    dataset = np.load('Actual/dataframe.npy', allow_pickle=True)
    #labels = np.load('Actual/labels.npy', allow_pickle=True)
    #full_board = np.load('Actual/data_with_walls.npy', allow_pickle=True)
    
    
  
    #adjacent alien probs
    dataset[:,1] = adj[:,0].tolist()
    
    #adjacent crew probs
    dataset[:,2] = adj[:,1].tolist()

    
    
    temp = adj[:,1]
    ar1 = np.where(temp[:,0] == 0)
    ar2 = np.where(temp[:,1] == 0)
    ar3 = np.where(temp[:,2] == 0)
    ar4 = np.where(temp[:,3] == 0)
    bad_dat = np.intersect1d(ar1,ar2)
    bad_dat = np.intersect1d(bad_dat,ar3)
    bad_dat = np.intersect1d(bad_dat, ar4)
    

    
    dataset = np.delete(dataset, bad_dat, axis = 0)
    
    

    
    
    if real_data:
        
        X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)

        X_train[:,0] = X_train[:,0] / (30*30)
        X_test[:,0] = X_test[:,0] / (30*30)
        in_size = (len(dataset[0,1])*2)+1

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

        # # print(X_train.shape)
        # # print(X_test.shape)

        # in_size = k
        drop_out = 0.7
        #rint(in_size)
        #nn = NN(in_size, 1000, 900, 700, 550, 400, 150, 75, 25, 5, .1, 0.0001, 5, 719, model_type, distances_array, drop_out, True) #53, model_type)
        nn = NN(in_size, 80, 45, 30, 25, 20, 15, 12, 10, 5, .1, 0.000001, 10, 719, model_type, distances_array, drop_out, True) #53, model_type)
        #nn = NN(in_size, 100, 60, 45, 30, 25, 20, 15, 10, 5, .1, 0, 10000, 67, model_type, distances_array, drop_out, True) #53, model_type)
        
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
        nn = NN(1, 10, 9, 8, 7, 6, 5, 4, 3, 5, .05, 0, 2000, 53, model_type, distances_array, drop_out, False)

    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

def model2(distances_array,train_split,  real_data):
    model_type = 2
    dataset = np.load('Actual/dataframe.npy', allow_pickle=True)
    cont_lab = np.load('continuous_labels.npy', allow_pickle=True)
    dataset[:,-1] = np.squeeze(cont_lab)
    print(dataset[0:6,-1])
    #print(dataset[:,1].shape)
    #print(dataset[:,1][0].shape)
    
    #dataset[:,1] = np.asarray(adj[:,0].tolist(), dtype=object)
    #print("test1",dataset[:,1][0])
    #print("test2", adj[:,0][0])
    
    #adjacent alien probs
    #dataset[:,1] = adj[:,0].tolist()
    #dataset[:,1] = np.zeros((dataset.shape[0], 4)).tolist()
    
    #adjacent crew probs
    #dataset[:,2] = np.zeros((78249, 4)).tolist()
    
    #print(np.where(adj[:,1][0] == 0))
    #print("yes",np.where(adj[:,1][0] == 0 and adj[:,1][1] == 0 and adj[:,1][2] == 0 and adj[:,1][3] == 0))
    #dataset[:,2] = adj[:,1].tolist()
    
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

        #y_train = np.squeeze(y_train)
        #y_test = np.squeeze(y_test)


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
        nn = NN(in_size, 900, 800, 500, 200, 100, 50, 25, 15, 1, 0.1, 0.01, 2, 719, model_type, distances_array, drop_out, True) #53, model_type)
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
        nn = NN(in_size, 600, 500, 400, 300, 200, 100, 50, 25, 1, .01, 0.01, 1000, 53, model_type, distances_array, drop_out, False)
        
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

def model3(distances_array, num_epochs, shp):
    train_split = 0.7
    dataset = np.load('Actual/dataframe.npy', allow_pickle=True)
    adj = np.load('Actual/data_adjacent.npy', allow_pickle=True)
    #dataset = np.load('Actual/dataframe.npy', allow_pickle=True)
    #labels = np.load('Actual/labels.npy', allow_pickle=True)
    #full_board = np.load('Actual/data_with_walls.npy', allow_pickle=True)
    
    #adjacent alien probs
    dataset[:,1] = adj[:,0].tolist()
    
    #adjacent crew probs
    dataset[:,2] = adj[:,1].tolist()

    
    
    temp = adj[:,1]
    ar1 = np.where(temp[:,0] == 0)
    ar2 = np.where(temp[:,1] == 0)
    ar3 = np.where(temp[:,2] == 0)
    ar4 = np.where(temp[:,3] == 0)
    bad_dat = np.intersect1d(ar1,ar2)
    bad_dat = np.intersect1d(bad_dat,ar3)
    bad_dat = np.intersect1d(bad_dat, ar4)
    
    dataset = np.delete(dataset, bad_dat, axis = 0)

    
    #return 0
    model_type = 1
    drop_out = 0.7
    
    # Simulate initial board states
    X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)


    X_train[:,0] = X_train[:,0] / (30*30)
    X_test[:,0] = X_test[:,0] / (30*30)
    in_size = (len(dataset[0,1])*2)+1

    avg_success_rate_T = []
    avg_trial_len_T = []
    avg_success_trial_len_T = []
    
    #Train MODEL 1 HERE 
    nn_actor = NN(9, 80, 45, 30, 25, 20, 15, 12, 10, 5, .1, 0.001, 5, 719, 1, distances_array, drop_out, True) 
    nn_actor.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    print("ACTOR FINISHED")
    model3_data, labels, avg_success_rate, avg_trial_len, avg_success_trial_len =  nn_bot_data(nn_actor, shp)
    avg_success_rate_T.append(avg_success_rate)
    avg_trial_len_T.append(avg_trial_len)
    avg_success_trial_len_T.append(avg_success_trial_len)

    num_rows = len(model3_data)
    model3_data = np.asarray(model3_data)
    model3_data = model3_data.reshape(num_rows, 10)
    labels = labels.T

    dataset = np.hstack((model3_data, labels.reshape(-1,1)))

    boundary = int(math.ceil(dataset.shape[0]*train_split))

    rng = default_rng()
    idx = rng.choice(dataset.shape[0], size = boundary, replace=False)
    mask = np.ones(dataset.shape[0], dtype=bool)
    mask[idx] = False

    train = dataset[~mask, :]
    test = dataset[mask, :]

    X_train = train[:,:-1]
    X_test = test[:,:-1]

    y_train = train[:,-1]
    y_train = y_train.reshape(-1, 1)
    y_test = test[:,-1]
    y_test = y_test.reshape(-1, 1)
    print(X_train.shape)
    print(y_train.shape)
    

    #Maybe we need to unsqueeze this?

    # Train MODEL 2 HERE
    nn_critic = NN(10, 80, 45, 30, 25, 20, 15, 10, 5, 1, .1, 0.01, 5, 67, 2, distances_array, drop_out, True)
    nn_critic.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    print("CRITIC FINISHED")
    
    

    # Repeat:
    for epoch in range(num_epochs):
        # Update model 1 using labels from model 2 instead of actual success labels
        print("inside epic loop")
        for elem in model3_data:
            prob = []
            for i in range(5):
                #print(i)
                elem[9] = i
                prob.append(nn_critic.predict(elem, True))
            elem[9] = np.argmax(np.asarray(prob))
        print("out of first loop")
        one_hot = np.zeros((num_rows, 5))
        for i in range(model3_data.shape[0]):
            one_hot[i,model3_data[i,9]] = 1
        model3_data = np.delete(model3_data, [-1], axis=1)
        model3_data = np.hstack((model3_data, one_hot))
        
        print("after 1 hot added", model3_data.shape)

        boundary = int(math.ceil(model3_data.shape[0]*train_split))

        rng = default_rng()
        idx = rng.choice(model3_data.shape[0], size = boundary, replace=False)
        mask = np.ones(model3_data.shape[0], dtype=bool)
        mask[idx] = False

        train = model3_data[~mask, :]
        test = model3_data[mask, :]

        X_train = train[:,:-5]
        X_test = test[:,:-5]

        y_train = train[:,-5:]
        y_test = test[:,-5:]
        
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        
        nn_actor.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
        model3_data, labels, avg_success_rate, avg_trial_len, avg_success_trial_len =  nn_bot_data(nn_actor, shp)
        avg_success_rate_T.append(avg_success_rate)
        avg_trial_len_T.append(avg_trial_len)
        avg_success_trial_len_T.append(avg_success_trial_len)

        num_rows = len(model3_data)
        model3_data = np.asarray(model3_data)
        model3_data = model3_data.reshape(num_rows, 10)
        labels = labels.T

        dataset = np.hstack((model3_data, labels.reshape(-1,1)))

        boundary = int(math.ceil(dataset.shape[0]*train_split))

        rng = default_rng()
        idx = rng.choice(dataset.shape[0], size = boundary, replace=False)
        mask = np.ones(dataset.shape[0], dtype=bool)
        mask[idx] = False

        train = dataset[~mask, :]
        test = dataset[mask, :]

        X_train = train[:,:-1]
        X_test = test[:,:-1]

        y_train = train[:,-1]
        y_train = y_train.reshape(-1, 1)
        y_test = test[:,-1]
        y_test = y_test.reshape(-1, 1)
        nn_critic.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
        
    # Run a bot with model 1.
    plt.plot(avg_success_rate_T)
    plt.title('Average Success Rate of Actor Network Bot over Time')
    plt.xlabel('Number of training iterations by Critic Network')
    plt.ylabel('Average Success Rate of Actor Network')
    plt.savefig('Success_Rate_AC.png')
    plt.show()
    plt.close()

    plt.plot(avg_trial_len_T)
    plt.title('Average Trial Length of Actor Network Bot over Time')
    plt.xlabel('Number of training iterations by Critic Network')
    plt.ylabel('Average Trial Length')
    plt.savefig('Trial_Length_AC.png')
    plt.show()
    plt.close()

    plt.plot(avg_success_trial_len_T)
    plt.title('Average Length of Successful Trial of Actor Network Bot over Time')
    plt.xlabel('Number of training iterations by Critic Network')
    plt.ylabel('Average Length of Successful Trial')
    plt.savefig('Succ_Trial_Length_AC.png')
    plt.show()
    plt.close()

    compareBots(nn_actor, shp)


def simulateData(k,boards):
    """This runs the 1 alien, 1 crew member experiments"""
    #numBoards = len(boards)
    numTrials = 1000
    #bots = [1]
    success_flag = False
    
    shp = boards[0]
    alpha = 0.5
    data = []
    data_with_walls = []
    data_sensor = []
    data_adjacent = []
    data_bot_loc = []
    data_bot_loc_open = []
    labels = []

    board_walls = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            if not shp.ship[i,j].is_open():
                board_walls[i,j] = -1

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
        #np.save('init_crew_probs.npy', shp.get_crew_probs())
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
            datarow_walls = []
            datarow_ds = []
            datarow_adj = []
            #if T > 6000:
            #    print(f"TIMEOUT: {T}")
            #    labels.append(np.zeros(T+1))
            #    flag = False
            #    break
            datarow.append(shp.ship[bot.row, bot.col].one_d_idx)
            tmp = np.copy(board_walls)
            tmp[bot.row, bot.col] = 1
            data_bot_loc.append(tmp) 
            data_bot_loc_open.append(tmp[shp.open_cell_indices()].flatten())

            ap = shp.get_alien_probs()
            cp = shp.get_crew_probs()

            datarow_walls.append(ap)
            datarow_walls.append(cp)
            data_with_walls.append(datarow_walls)

            

            ap1 = ap[shp.open_cell_indices()].flatten()
            cp1 = cp[shp.open_cell_indices()].flatten()

            datarow.append(ap1)
            datarow.append(cp1)

            r, c = shp.get_det_sq_indicies()
            ap2 = ap[r, c].flatten()
            cp2 = cp[r, c].flatten()

            datarow_ds.append(ap2)
            datarow_ds.append(cp2)
            data_sensor.append(datarow_ds)

            ap_vector = np.ones(4)
            cp_vector = np.zeros(4)

            if bot.col > 0:
                if shp.ship[bot.row, bot.col-1].is_open():
                    ap_vector[0] = ap[bot.row, bot.col-1]
                    cp_vector[0] = cp[bot.row, bot.col-1]
                else:
                    ap_vector[0] = 1.0
                    cp_vector[0] = 0.0
            if bot.col < 29:
                if shp.ship[bot.row, bot.col+1].is_open():
                    ap_vector[1] = ap[bot.row, bot.col+1]
                    cp_vector[1] = cp[bot.row, bot.col+1]
                else:
                    ap_vector[1] = 1.0
                    cp_vector[1] = 0.0
            if bot.row > 0:
                if shp.ship[bot.row-1, bot.col].is_open():
                    ap_vector[2] = ap[bot.row-1, bot.col]
                    cp_vector[2] = cp[bot.row-1, bot.col]
                else:
                    ap_vector[2] = 1.0
                    cp_vector[2] = 0.0
            if bot.row < 29:
                if shp.ship[bot.row+1, bot.col].is_open():
                    ap_vector[3] = ap[bot.row+1, bot.col]
                    cp_vector[3] = cp[bot.row+1, bot.col]
                else:
                    ap_vector[3] = 1.0
                    cp_vector[3] = 0.0
            
            datarow_adj.append(ap_vector)
            datarow_adj.append(cp_vector)
            data_adjacent.append(datarow_adj)
            

             

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

    np.save('Actual/data.npy', data)
    np.save('Actual/labels.npy', labels)
    
    np.save('Actual/data_with_walls.npy', data_with_walls)
    np.save('Actual/data_sensor.npy', data_sensor)
    np.save('Actual/data_adjacent.npy', data_adjacent)
    np.save('Actual/data_bot_loc.npy', data_bot_loc)
    np.save('Actual/data_bot_loc_open.npy', data_bot_loc_open)
    dataframe = np.hstack((data, labels))
    np.save('Actual/dataframe.npy', dataframe)

    


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
    with open('Actual/board.pickle', "wb") as b_file:
        pickle.dump(shp, b_file, pickle.HIGHEST_PROTOCOL)
    np.save('Actual/board.npy', boards[0].ship)
    simulateData(k, boards)

def balanceSet(X_data, y_data, model_type):

    X_D = np.hstack((X_data, y_data))
    X_D = X_D.astype('float')
    print("DIVIDER")
    #print(X_D[:,-5])
    if model_type == 1:
        L = np.bincount(X_D[:,-5].astype('int'))
        print("left:", L)
        Lii = np.nonzero(L)[0]
        Lfreq = np.vstack((Lii,L[Lii])).T
        
        R = np.bincount(X_D[:,-4].astype('int'))
        print("right:", R)
        Rii = np.nonzero(R)[0]
        Rfreq = np.vstack((Rii,R[Rii])).T

        U = np.bincount(X_D[:,-3].astype('int'))
        print("up:", U)
        Uii = np.nonzero(U)[0]
        Ufreq = np.vstack((Uii,U[Uii])).T

        D = np.bincount(X_D[:,-2].astype('int'))
        print("down:", D)
        Dii = np.nonzero(D)[0]
        Dfreq = np.vstack((Dii,D[Dii])).T
        
        X = np.bincount(X_D[:,-1].astype('int'))
        print("stay:", X)
        Xii = np.nonzero(X)[0]
        Xfreq = np.vstack((Xii,X[Xii])).T
        
        freqs = [Lfreq[1,:], Rfreq[1,:], Ufreq[1,:], Dfreq[1,:], Xfreq[1,:]]
        freqs = np.asarray(freqs)
        idx = np.argmin(freqs[:,1])
        lower_bound = freqs[idx][1]
        print(idx)
        print(lower_bound)

        idxes = [0, 1, 2, 3, 4]
        idxes.remove(idx)
        for i in range(len(idxes)):
            idxes[i] = idxes[i] - 5
        print(idxes)
        samples = []
        for i in idxes:
            subset = X_D[X_D[:,i] == 1]
            k = subset[np.random.choice(subset.shape[0], lower_bound, replace=False), :]
            print(k.shape)
            samples.append(k)
        print(len(samples))
        F = X_D[X_D[:,idx-5] == 1]
        print(F.shape)
        for samp in samples:
            F = np.vstack((F, samp))
            print(F.shape)
        
        X_data = np.delete(F, [-1], axis=1)
        y_data = F[:,-5:]
        X_data = np.delete(F, [-5, -4, -3, -2, -1], axis=1)
        print()
        print()
        print()
        print('FINAAL:', X_data.shape)
        print(y_data.shape)
        print()
        return (X_data, y_data)
    else:
        y = np.bincount(X_D[:,-1].astype('int'))
        ii = np.nonzero(y)[0]
        freq = np.vstack((ii,y[ii])).T
        idx = np.argmin(freq[:,1])
        val = np.argmin(freq[:,0])
        lower_bound = freq[idx,1]
        q = X_D[X_D[:, -1].astype('int') == val]
        p = X_D[X_D[:, -1].astype('int') == abs(val - 1)]
        q = q[np.random.choice(q.shape[0], lower_bound, replace=False), :]

        X = np.vstack((p, q))
        y_data = X[:,-1].astype('int')
        y_data = y_data.reshape(-1, 1)
        X_data = np.delete(X, [-1], axis=1)
        
        return (X_data, y_data)


def compareBots(model, shp):
    """This runs bot 1 with and without neural network"""
    numTrials = 50
    k = 3
    avg_trial_len = np.zeros((2))
    avg_success_trial_len = np.zeros((2))
    avg_success_rate = np.zeros((2))
    
    success_flag = False
    
    shp1 = shp
    shp2 = copy.deepcopy(shp)
    alpha = 0.5
    model3_data = []

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
            
            if T > 3000:
                print(f"Timeout: {T}")
                avg_trial_len[0] += T
                flag = False
                break

            if shp1.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                avg_trial_len[0] += T
                flag = False
                break
            if shp1.ship[i][j].contains_crew():
                print(f"Saved: {T}")
                avg_success_trial_len[0] += T
                avg_success_rate[0] += 1
                avg_trial_len[0] += T
    
                shp1.ship[i][j].remove_crew()
                flag = False
                break
            shp1.one_one_bot_move_update()
            if alien1.move():
                print(f"Dead: {T}")
                avg_trial_len[0] += T / (numTrials)
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
            
            #print(cur_board_state)
            pred = model.predict(cur_board_state, True)
            move = bot2.nn_bot_move(pred)
            
            np.append(cur_board_state, move)
            model3_data.append(cur_board_state)
            
            # MUST RECORD THE MOVE IT TAKES
            i = bot2.row
            j = bot2.col
            
            if T > 3000:
                print(f"Timeout: {T}")
                avg_trial_len[1] += T
                flag = False
                break

            if shp2.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                avg_trial_len[1] += T
                flag = False
                break
            if shp2.ship[i][j].contains_crew():
                print(f"Saved: {T}")
                avg_success_trial_len[1] += T
                avg_success_rate[1] += 1
                avg_trial_len[1] += T
                shp2.ship[i][j].remove_crew()
                flag = False
                break

            shp2.one_one_bot_move_update()
            #print("bot move: ", shp.get_crew_probs())

            if alien2.move():
                print(f"Dead: {T}")
                avg_trial_len[1] += T
                flag = False
                break
            shp2.one_one_alien_move_update()
                        
            alien_beep = bot2.detect_alien()
            shp2.one_one_alien_beep_update(alien_beep)
            crew_beep = bot2.new_detect_crew(start_cells[0], start_cells[1])
            shp2.one_one_crew_beep_update(crew_beep)
            T += 1
        shp2.empty_ship()
    avg_success_trial_len[0] /= avg_success_rate[0]
    avg_success_trial_len[1] /= avg_success_rate[1]
    avg_success_rate[0] /= numTrials
    avg_success_rate[1] /= numTrials
    avg_trial_len[0] /= numTrials
    avg_trial_len[1] /= numTrials


    fig, ax = plt.subplots()

    fruits = ['Bot 1 (Default)', 'Bot Using Neural Network']
    counts = [avg_success_rate[0], avg_success_rate[1]]
    bar_labels = ['Bot 1 (Default)', 'Bot Using Neural Network']
    bar_colors = ['tab:red', 'tab:blue']

    ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('Success Rate')
    ax.set_title('Trial Success Rate of Bot 1 and Bot NN')
    ax.legend(title='Bot Type')

    plt.savefig('success_prob.png')
    plt.show()
    plt.close()

    group = ("Average Success Rate", "Average Length of Successful Trial", "Average Length of Trial")
    group_means = {
        "Bot 1 (Default)": (avg_success_trial_len[0], avg_trial_len[0]),
        "Bot Using Neural Network": (avg_success_trial_len[1], avg_trial_len[1])
    }
    z = np.arange(len(group))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in group_means.items():
        offset = width * multiplier
        rects = ax.bar(z + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Length of Trial')
    ax.set_title('Average Length of Trials Bot 1 and Bot NN')
    ax.set_xticks(z + width, group)
    ax.legend(loc='upper left', ncols=3)
    
    plt.savefig('trial_length_plot.png')
    plt.show()
    plt.close()
    
def nn_bot_data(model, shp):
    """This runs bot 1 with and without neural network"""
    numTrials = 2
    k = 3
    avg_trial_len = 0
    avg_success_trial_len = 0
    avg_success_rate = 0
    
    success_flag = False
    
    shp1 = shp
    shp2 = copy.deepcopy(shp)
    alpha = 0.5
    model3_data = []
    labels = []

    for trial in range(numTrials):
       
        i, j = shp1.get_unoccupied_cell()
        bot2 = Bot(i, j, k, shp2, 2, alpha)
        shp2.bot = bot2

        start_cells = []
        i, j = shp1.get_unoccupied_cell()
        shp2.ship[i][j].add_crew()
        start_cells.append(i)
        start_cells.append(j)

        i, j = shp1.get_unoccupied_alien_cell(k)
        alien2 = Alien(i, j, shp2)

        shp2.distances_from_crew()
        # print("Calculated distances")
   
        shp2.init_crew_prob_one()
                    
        shp2.init_alien_prob_one()

        
        print('Trial:', trial)

        #Run bot2
        T = 0
        flag = True
        while flag:
            
            # Neural Network powered bot move sequence
            cur_board_state = bot2.get_cur_state()
            
            #print(cur_board_state)
            pred = model.predict(cur_board_state, True)
            move = bot2.nn_bot_move(pred)
            
            cur_board_state = np.append(cur_board_state,move)
            model3_data.append(cur_board_state)
            
            # MUST RECORD THE MOVE IT TAKES
            i = bot2.row
            j = bot2.col
            
            if T > 3000:
                print(f"Timeout: {T}")
                avg_trial_len += T
                flag = False
                labels.append(np.zeros(3002))
                break

            if shp2.ship[i][j].contains_alien():
                print(f"Dead: {T}")
                avg_trial_len += T
                flag = False
                labels.append(np.zeros(T+1))
                break
            if shp2.ship[i][j].contains_crew():
                print(f"Saved: {T}")
                avg_success_trial_len += T
                avg_success_rate += 1
                avg_trial_len += T
                shp2.ship[i][j].remove_crew()
                labels.append(np.zeros(T+1))
                flag = False
                break

            shp2.one_one_bot_move_update()
            #print("bot move: ", shp.get_crew_probs())

            if alien2.move():
                print(f"Dead: {T}")
                avg_trial_len += T
                flag = False
                labels.append(np.zeros(T+1))
                break
            shp2.one_one_alien_move_update()
                        
            alien_beep = bot2.detect_alien()
            shp2.one_one_alien_beep_update(alien_beep)
            crew_beep = bot2.new_detect_crew(start_cells[0], start_cells[1])
            shp2.one_one_crew_beep_update(crew_beep)
            T += 1
        shp2.empty_ship()
    if avg_success_rate != 0:
        avg_success_trial_len /= avg_success_rate
    avg_success_rate /= numTrials
    avg_trial_len /= numTrials

    print(len(labels))
    print(len(model3_data))
    labels = np.concatenate(labels, axis=None).ravel()
    print(len(labels))
    return model3_data, labels, avg_success_rate, avg_trial_len, avg_success_trial_len

    
if __name__ == "__main__":
    #x = np.load('Final/board.npy', allow_pickle=True)
    # open a file, where you stored the pickled data

    
    file = open('Actual/board.pickle', 'rb')
    shp = pickle.load(file)
    #shp.print_ship()
    
    file.close()
    shp.open_cell_indices()
    shp.open_cell_distances()
    #print(shp.print_ship())
    distances = shp.get_one_distances()
    #print(distances.shape)

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
    #nn = model1(distances,train_split=0.7,  real_data = False)
    
    #nn = model2(distances,train_split=0.7,  real_data = True)
    #nn = model2(distances,train_split=0.7,  real_data = False)
    #dat = nn_bot_data(nn, shp)
    #compareBots(nn, shp)
    model3(distances, 2, shp)


    #model3(distances,=0.7, num_epochs=100, shp)
    
    