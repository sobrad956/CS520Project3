import numpy as np
import random
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot


class NN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, model_type):
        np.random.seed(0)
        self.input_size = input_size
        self.hidden_size = hidden_size #only did 1 hidden layer so far
        self.output_size = output_size
        self.lr = learning_rate  #Haven't done anything smarter than standard SGD here
        self.epochs = num_epochs
        self.batch_size = batch_size 
        self.model_type = model_type  #Have to define differences in gradients in backprop function
        

    # initialize weights and biases
    def initialize_parameters(self):
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01 #Haven't done anything interesting in initialization
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.output_size, 1))
    
    #Activation & Loss Functions
    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
    
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
        return(np.exp(x)/np.exp(x).sum())
    
    def binary_cross_entropy_loss(self, y_pred, y_true): #For binary classification
        m = y_true.shape[0]
        return -(1/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    
    def cross_entropy_loss(self, y_pred, y_true): #For multiclass classification
        m = y_true.shape[0]
        return -(1/m) * np.sum(y_true * np.log(y_pred.T + 10**-100)) #Added small error for divide by zero errors
    
    # forward propagation
    def forward_propagation(self, X):    
        # compute the activation of the hidden layer
        self.Z1 = np.matmul(self.W1, X.T) + self.b1
        self.A1 = self.sigmoid(self.Z1)
    
        # compute the activation of the output layer
        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        if self.model_type == 1:
            self.A2 = self.softmax(self.Z2)
        else:
            self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    # backward propagation
    def backward_propagation(self, X, y):
        #pass
        #y = y.T
        m = y.shape[0]
        
        # compute the derivative of the loss with respect to A2
        #print(y.shape)
        #print(self.A2.shape)
        #dA2 = - (y/self.A2) + ((1-y)/(1-self.A2))
    
        # compute the derivative of the activation function of the output layer
        #dZ2 = dA2 * (self.A2 * (1-self.A2))
    
        # compute the derivative of the weights and biases of the output layer
        #self.dW2 = (1/m) * np.matmul(dZ2, self.A1.T)
        #self.db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
        # # compute the derivative of the activation function of the hidden layer
        #dA1 = np.dot(self.W2.T, dZ2)
        #dZ1 = dA1 * (self.A1 * (1-self.A1))
    
        # # compute the derivative of the weights and biases of the hidden layer
        #self.dW1 = (1/m) * np.dot(dZ1, X)
        #self.db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # update parameters
    def update_parameters(self):
        # update the weights and biases
        self.W1 = self.W1 - self.lr * self.dW1
        self.b1 = self.b1 - self.lr * self.db1
        self.W2 = self.W2 - self.lr * self.dW2
        self.b2 = self.b2 - self.lr * self.db2
    
    # train the neural network
    def train(self, X, y):
        # initialize the weights and biases
        self.initialize_parameters()
    
        for i in range(self.epochs):
            batch_indices = random.sample([i for i in range(X.shape[0])], k = self.batch_size)
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]
            # forward propagation
            self.forward_propagation(x_batch)
        
            # compute the loss
            if self.model_type == 1:
                loss = self.cross_entropy_loss(self.A2, y_batch)
            else:
                loss = self.binary_cross_entropy_loss(self.A2, y_batch)
        
            # backward propagation
            self.backward_propagation(x_batch, y_batch)
        
            # update the parameters
            self.update_parameters()
        
            if i % 10 == 0:
                print(f"iteration {i}: loss = {loss}")
    
    # predict the labels for new data
    #def predict(self, X):
    #    self.forward_propagation(X)
    #    predictions = (self.A2 > 0.5).astype(int)
    #    return predictions


def clean_data(dataset, train_split, model_type):
    boundary = int(math.ceil(dataset.shape[0]*train_split))
    train = dataset[:boundary]
    test = dataset[boundary:]

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
    HH = np.hstack((np.concatenate(X_train[:,1]), np.concatenate(X_train[:,2]))).reshape(X_train.shape[0], 1252)
    X_train = np.hstack((X_train[:,0].reshape(-1, 1), HH))
    print(X_train.shape)
    if model_type == 1:
        y_train = np.concatenate(y_train[:,0]).reshape(X_train.shape[0], 5)
    else:
        y_train = y_train[:,0].reshape(X_train.shape[0], 1)
    print(y_train.shape)

    print()

    print("Test Reshaped:")
    HH = np.hstack((np.concatenate(X_test[:,1]), np.concatenate(X_test[:,2]))).reshape(X_test.shape[0], 1252)
    X_test = np.hstack((X_test[:,0].reshape(-1, 1), HH))
    print(X_test.shape)
    if model_type == 1:
        y_test = np.concatenate(y_test[:,0]).reshape(X_test.shape[0], 5)
    else: 
        y_test = y_test[:,0].reshape(X_test.shape[0], 1)
    print(y_test.shape)
    return(X_train, y_train, X_test, y_test)


def model1(train_split):
    model_type = 1
    dataset = np.load('dataframe1.npy', allow_pickle=True)
    X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)

    nn = NN(1253, 300, 5, 0.01, 1000, 53, model_type)
    nn.train(X_train.astype(float), y_train.astype(float))

def model2(train_split):
    model_type = 2
    dataset = np.load('dataframe1.npy', allow_pickle=True)
    X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)

    nn = NN(1253, 300, 1, 0.01, 1000, 53, model_type) #Batch Size probably not 53 for whole dataset.
    nn.train(X_train.astype(float), y_train.astype(float))

def simulateData(k,boards):
    """This runs the 1 alien, 1 crew member experiments"""
    #numBoards = len(boards)
    numTrials = 1000
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
        shp = Ship(k, 30)
        shp.generate_ship()
        print("ship generated")
        boards.append(shp)
    #experiement takes k, boards
    simulateData(k, boards)

    
if __name__ == "__main__":
    #runSimulate()
    #model1(train_split=0.7)
    model2(train_split=0.7)
    