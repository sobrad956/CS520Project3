import numpy as np
import random
from numpy.random import default_rng
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot
import pickle



class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate, num_epochs, batch_size, model_type):
        #np.random.seed(0)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
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

    # initialize weights and biases
    def initialize_parameters(self):
        self.W1 = np.random.randn(self.hidden_size1, self.input_size) * math.sqrt(1.0/self.input_size) #np.zeros(self.hidden_size, self.input_size) #np.random.randn(self.hidden_size, self.input_size) * 0.01 #Haven't done anything interesting in initialization
        self.b1 = np.zeros((self.hidden_size1, 1))
        self.W2 = np.random.randn(self.hidden_size2, self.hidden_size1) * math.sqrt(1.0/self.hidden_size1) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b2 = np.zeros((self.hidden_size2, 1))
        self.W3 = np.random.randn(self.output_size, self.hidden_size2) * math.sqrt(1.0/self.hidden_size2) #np.zeros(self.output_size, self.hidden_size) #np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b3 = np.zeros((self.output_size, 1))
    
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
        return(np.exp(x)/np.exp(x).sum())
    
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
        A1 = self.sigmoid(Z1)
        #self.A1 = self.relu(self.Z1)
    
        # compute the activation of the output layer
        #print("before layer 2")
        Z2 = np.matmul(self.W2, A1) + self.b2
        #if self.model_type == 1:
        #   A2 = self.softmax(Z2)
        #else:
        A2 = self.sigmoid(Z2)
        #self.A2 = self.relu(self.Z2)
        
        Z3 = np.matmul(self.W3, A2) + self.b3
        
        A3 = self.sigmoid(Z3)

        if test == False:
            self.Z1 = Z1
            self.A1 = A1
            self.Z2 = Z2
            self.A2 = A2
            self.Z3 = Z3
            self.A3 = A3
        
        return A3

    
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
            dA3 = (self.A3 - y) #sus
        else:
            dA3 = - (y/(self.A3)+1e-15) + ((1-y)/((1-self.A3)+1e-15)) #This is correct

        #print('dA2 shape: ', dA2.shape)
    
        # compute the derivative of the activation function of the output layer
        #dZ2 = dA2 * self.d_relu(self.Z2)
        dZ3 = dA3 * (self.A3 * (1-self.A3))
        
        self.dW3 = (1/m) * np.dot(dZ3, self.A2.T)
        self.db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
        

        #dZ2 = dA2 * self.d_sigmoid(self.Z2)
        dA2 = np.dot(self.W3.T, dZ3)

        dZ2 = dA2 * (self.A2 * (1-self.A2)) #This should be correct
        #print('dZ2 shape: ', dZ2.shape)
        #print('Z2 shape: ', self.Z2.shape)
    
        # compute the derivative of the weights and biases of the output layer
        self.dW2 = (1/m) * np.matmul(dZ2, self.A1.T)
        #print('dW2 shape: ', self.dW2.shape)
        #print('W2 shape: ', self.W2.shape)
        self.db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        #print('db2 shape: ', self.db2.shape)
        #print('b2 shape: ', self.b2.shape)
    
        # # compute the derivative of the activation function of the hidden layer
        dA1 = np.dot(self.W2.T, dZ2)
        #dZ1 = dA1 * self.d_sigmoid(self.Z1)
        #dZ1 = dA1 * self.d_relu(self.Z1)

        
        dZ1 = dA1 * (self.A1 * (1-self.A1))
        
        # print("dz1")
        # print(np.min(dZ1))
        # print()
        # # compute the derivative of the weights and biases of the hidden layer
        self.dW1 = (1/m) * np.dot(dZ1, X)
        self.db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # update parameters
    def update_parameters(self):
        # update the weights and biases
        self.W1 = self.W1 - self.lr * self.dW1
        self.b1 = self.b1 - self.lr * self.db1
        self.W2 = self.W2 - self.lr * self.dW2
        self.b2 = self.b2 - self.lr * self.db2
        self.W3 = self.W3 - self.lr * self.dW3
        self.b3 = self.b3 - self.lr * self.db3
        
    
    def zero_grad(self):
        self.dW1 = 0
        self.dW2 = 0
        self.dW3 = 0
        self.db1 = 0 
        self.db2 = 0
        self.db3 = 0
        
    
    # train the neural network


    def acc_score(self, y_actual, y_pred):
        if self.model_type == 1:
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
    
        for i in range(self.epochs):
        #for i in range(1):
            batch_indices = random.sample([i for i in range(X.shape[0])], k = self.batch_size)
            test_batch_indices = random.sample([i for i in range(X_test.shape[0])], k = self.batch_size)
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]

            x_test_batch = X_test[test_batch_indices]
            y_test_batch = y_test[test_batch_indices]
            #print('actual', y_test_batch[0:2,:])

            #x_batch = X
            #y_batch = y

            # forward propagation
            #print("before forward prob")
            self.forward_propagation(x_batch)

            
            y_batch = y_batch.T
            y_test_batch = y_test_batch.T
        
            # compute the loss
            #print("before loss compute")

            

            

            self.train_accuracies.append(self.acc_score(y_batch, self.A3))
            loss = self.cross_entropy_loss(self.A3, y_batch)
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
            self.test_accuracies.append(self.acc_score(y_test_batch, self.predict(x_test_batch)))

            test_loss = self.cross_entropy_loss(self.forward_propagation(x_test_batch, True), y_test_batch)
            

            self.test_losses_smooth.append(test_loss)
            self.train_losses_smooth.append(loss)

            if i % 10 == 0:
                
                # self.predict(x_test_batch)
                #test_loss = self.binary_cross_entropy_loss(self.predict(X_test), y_test.T)
                #print(f"iteration {i}: Total train loss = {loss}")
                loss_smooth = np.mean(np.asarray(self.train_losses_smooth))
                test_loss_smooth = np.mean(np.asarray(self.test_losses_smooth))

                self.test_losses.append(test_loss_smooth)
                self.train_losses.append(loss_smooth)

                print(f"iteration {i}: Total train loss = {loss_smooth}, total test loss = {test_loss_smooth}")
                # print(f"iteration {i}: train loss = {loss}, test loss = {test_loss}")
                self.test_losses_smooth = []
                self.train_losses_smooth = []
                

            #print(self.A2)
            #print(self.b1)

    
    # predict the labels for new data
    def predict(self, X):
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
    HH = np.hstack((np.concatenate(X_train[:,1]), np.concatenate(X_train[:,2]))).reshape(X_train.shape[0], 1252)
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
    HH = np.hstack((np.concatenate(X_test[:,1]), np.concatenate(X_test[:,2]))).reshape(X_test.shape[0], 1252)
    X_test = np.hstack((X_test[:,0].reshape(-1, 1), HH))
    print(X_test.shape)
    if model_type == 1:
        y_test = np.concatenate(y_test[:,0]).reshape(X_test.shape[0], 5)
    else: 
        y_test = y_test[:,0].reshape(X_test.shape[0], 1)
    print(y_test.shape)
    return(X_train, y_train, X_test, y_test)


def model1(train_split, real_data):
    model_type = 1
    dataset = np.load('dataframe1.npy', allow_pickle=True)
    
    if real_data:
        
        X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)
        X_train[:,0] = X_train[:,0] / (30*30)
        X_test[:,0] = X_test[:,0] / (30*30)
        in_size = (len(dataset[0,1])*2)+1
        nn = NN(in_size, 150, 50, 5, 0.5, 1000, 53, model_type)
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
        
        nn = NN(1, 20, 10, 5, .1 ,1000, 53, model_type)

    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

def model2(train_split, real_data):
    model_type = 2
    dataset = np.load('dataframe1.npy', allow_pickle=True)
    
    if real_data:
        X_train, y_train, X_test, y_test = clean_data(dataset, train_split, model_type)
        X_train[:,0] = X_train[:,0] / (30*30)
        X_test[:,0] = X_test[:,0] / (30*30)
        in_size = (len(dataset[0,1])*2)+1

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        nn = NN(in_size, 600, 100, 1, .1, 1000, 53, model_type) #Batch Size probably not 53 for whole dataset.
    else:
        X_train = np.random.randn(59042, 5) # matrix of random x data
        y_train = X_train[:,1] > 0.5

        X_test = np.random.randn( 25303, 5) # matrix of random x data
        y_test = X_test[:,1] > 0.5
        y_test = y_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        print(y_test.shape)
        print(y_train.shape)
        nn = NN(5, 20,10, 1, .1, 1000, 53, model_type)
        
    nn.train(X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float))
    nn.plotLoss()
    nn.plotAcc()
    return nn

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
        pickle.dump(boards[0], b_file, pickle.HIGHEST_PROTOCOL)
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
    #runSimulate()
    nn = model1(train_split=0.7, real_data = True)
    #nn = model1(train_split=0.7, real_data = False)
    
    #nn = model2(train_split=0.7, real_data = True)
    #nn = model2(train_split=0.7,real_data = False)
    #compareBots(nn)
    
    