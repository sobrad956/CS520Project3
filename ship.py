import random
import math
import numpy as np
from queue import Queue
import copy
from itertools import product
from itertools import combinations, combinations_with_replacement
import bot
import pickle


class Cell:
    """ This class is used to record the state of a cell on the ship and any occupants on the cell """
    
    def __init__(self, row, col, d):
        """ By default, a cell is closed and nothing occupies it """
        self.state = '#'
        self.row = row
        self.col = col
        self.alien = False
        self.crew = False
        self.bot = False
        self.d = d  # Size of the ship
        self.distances = np.asarray([[-1 for j in range(self.d)] for i in range(self.d)])  # Stores distance from this cell to all other cells in the board, closed cells have distance of -1
        self.open_n = []  # Stores the open cells that are directly adjacent to this cell
        self.one_d_idx = -1
        
    def set_distance(self, row, col, dist):
        self.distances[row][col] = dist

    def get_location(self):
        return self.row, self.col

    def get_state(self):
        return self.state
    
    def is_open(self):
        return self.state == 'O'
    
    def open_cell(self):
        self.state = 'O'
        
    def close_cell(self):
        self.state = '#'

    def contains_bot(self):
        return self.bot

    def add_bot(self):
        self.bot = True

    def remove_bot(self):
        self.bot = False

    def contains_alien(self):
        return self.alien
    
    def add_alien(self):
        self.alien = True

    def remove_alien(self):
        self.alien = False

    def contains_crew(self):
        return self.crew

    def add_crew(self):
        self.crew = True

    def remove_crew(self):
        self.crew = False


class Ship:
    """ This class is used to arrange cells in a grid to represent the ship and generate it at time T=0 """
    
    def __init__(self, shp_path = None):
        self.D = 30  # The dimension of the ship as a square
        if shp_path is None:
            self.ship = np.asarray([[Cell(i, j, self.D) for j in range(self.D)] for i in range(self.D)])  # creates a DxD 2D grid of closed cells
        else:
            try:
                with open('board.pickle', "rb") as b_file:
                    ship_import = pickle.load(b_file)
            except FileNotFoundError:
                pass
            self.ship = ship_import.ship
            #self.ship = np.load(shp_path, allow_pickle=True)[0].ship

        #    print(self.ship.shape)
        self.bot_loc = [-1, -1]  # Stores the initial position of the bot, used to restrict alien generation cells
        self.crew_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])  # Stores the probability of a crew being in a cell for the ship
        self.alien_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)]) # Stores the probability of an alien being in a cell for the ship
        self.k = 3  # Size of detection square radius
        self.bot = None  # Reference to the bot on a given ship
        self.num_open_cells = None  # Number of open cells in the ship
        if shp_path != None:
            self.open_neighbors = ship_import.open_neighbors
        else:
            self.open_neighbors = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])  # stores the number of cells adjacent to a given cell that are open
        #self.two_crew_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])  # Stores the probability of a pair of crew being in two cells for the ship
        #self.two_alien_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)]) # Stores the probability of a pair of aliens being in two cells for the ship
        self.open_cell_mask = np.asarray([[False for j in range(self.D)] for i in range(self.D)])  # Mask for the location in the ship of all open cells
        #self.neighbor_pair_array = np.asarray([[[[None for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])  # For a given pair of open cells, stores all the possible pairs of neigboring cells to the input cells
        if shp_path != None:
            self.calculate_open_cells()
            idx = 0
            for i in range(self.D):
                for j in range(self.D):
                    if self.ship[i][j].is_open():
                        self.ship[i][j].one_d_idx = idx
                        idx += 1


    def get_crew_probs(self):
        return self.crew_probs

    def get_two_crew_probs(self):
        return self.two_crew_prob
    
    def get_alien_probs(self):
        return self.alien_probs
    
    def get_two_alien_probs(self):
        return self.two_alien_prob

    def set_crew_probs(self, i, j, p):
        self.crew_probs[i][j] = p

    def set_alien_probs(self, i, j, p):
        self.alien_probs[i][j] = p

    def set_bot_loc(self, i, j):
        self.bot_loc = [i, j]
    
    def get_bot_loc(self):
        return self.bot_loc

    def get_sensor_region(self, i, j):
        """ Returns only the cells in the ship that arre within the alien detection sensor region (square of side 2k + 1)"""
        return self.ship[max(i - self.k, 0):min(i + self.k + 1, self.D), max(j - self.k, 0):min(j + self.k + 1, self.D)]

    def print_ship(self):
        """ This function is used to visualize the current state of cells in the ship """
        for i in range(len(self.ship)):
            for j in range(len(self.ship[0])):
                if self.ship[i][j].contains_alien() and self.ship[i][j].contains_bot():
                    print('X', end=" ")
                else:
                    if self.ship[i][j].contains_alien() and self.ship[i][j].contains_crew():
                        print('D', end=" ")
                    else:
                        if self.ship[i][j].contains_alien():
                            print('a', end=" ")
                        else:
                            if self.ship[i][j].contains_crew():
                                print('c', end=" ")
                            else:
                                if self.ship[i][j].contains_bot():
                                    print('b', end=" ")
                                else:
                                    print(self.ship[i][j].get_state(), end=" ")
            print()
            
    def one_neighbor(self, i, j):
        """ If the given cell has more than 1 open neighbor, returns False, otherwise returns True """
        neighbors = 0

        if i > 0:  # If not in top row, check neighbor above
            up = self.ship[i-1][j]
            if up.is_open():
                neighbors += 1
        if i < self.D-1:  # If not in last row, check neighbor below
            down = self.ship[i+1][j]
            if down.is_open():
                neighbors += 1
        if j > 0:  # If not in left-most column, check neighbor to the left
            left = self.ship[i][j-1]
            if left.is_open():
                neighbors += 1
        if j < self.D-1:  # If not in right-most column, check neighbor to the right
            right = self.ship[i][j+1]
            if right.is_open():
                neighbors += 1
        self.open_neighbors[i][j] = neighbors
        if neighbors == 1:
            return True
        else:
            return False
            
    def closed_neighbors(self, i, j):
        """ Returns a list of closed neighbors for a given cell """
        neighbors = []  # keeps track of neighboring cells that are closed
        if i > 0 and not self.ship[i-1][j].is_open():
            neighbors.append((i-1, j))
        if i < self.D-1 and not self.ship[i+1][j].is_open():
            neighbors.append((i+1, j))
        if j > 0 and not self.ship[i][j-1].is_open():
            neighbors.append((i, j-1))
        if j < self.D-1 and not self.ship[i][j+1].is_open():
            neighbors.append((i, j+1))
        return neighbors
    
    def get_unoccupied_cell(self):
        """ Returns an unoccupied cell, crew members restricted from bot cell"""
        while True:
            i = random.randint(0, (self.D-1)) 
            j = random.randint(0, (self.D-1))
            if self.ship[i][j].is_open() and not self.ship[i][j].contains_bot() and not self.ship[i][j].contains_crew():
                return i, j

    def get_unoccupied_alien_cell(self, k):
        """ Returns an unoccupied cell, aliens restricted from detection square"""
        while True:
            i = random.randint(0, (self.D-1))
            j = random.randint(0, (self.D-1))

            if i < (self.bot_loc[0] - k) or i > (self.bot_loc[0] + k):
                if j < (self.bot_loc[1] - k) or j > (self.bot_loc[1] + k):
                    if self.ship[i][j].is_open() and not self.ship[i][j].contains_alien() and not self.ship[i][j].contains_bot():
                        return i, j

    def empty_ship(self):
        """ Resets the ship to default without generating new ship layout """
        self.crew_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.alien_probs = np.asarray([[0.0 for j in range(self.D)] for i in range(self.D)])
        self.two_crew_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])
        self.two_alien_prob = np.asarray([[[[0.0 for j in range(self.D)] for i in range(self.D)] for k in range(self.D)] for l in range(self.D)])

        self.bot = None
        for i in range(self.D):
            for j in range(self.D):
                if self.ship[i][j].contains_bot():
                    self.set_bot_loc(-1, -1)
                    self.ship[i][j].remove_bot()
                if self.ship[i][j].contains_alien():
                    self.ship[i][j].remove_alien()
                if self.ship[i][j].contains_crew():
                    self.ship[i][j].remove_crew()

    def calculate_open_cells(self):
        """ For every open cell in the ship, stores the neighboring open cells """
        for row in range(self.D):
            for col in range(self.D):
                if self.ship[row][col].is_open():
                    self.open_cell_mask[row][col] = True
                    if(row > 0):
                        if self.ship[row-1][col].is_open():
                            self.ship[row][col].open_n.append((row-1, col))
                    if(row < self.D -1):
                        if self.ship[row+1][col].is_open():
                            self.ship[row][col].open_n.append((row+1,col))
                    if(col > 0):
                        if self.ship[row][col-1].is_open():
                            self.ship[row][col].open_n.append((row,col-1))
                    if(col < self.D -1):
                        if self.ship[row][col+1].is_open():
                            self.ship[row][col].open_n.append((row, col+1))
                    

    def generate_ship(self):
        """ This function generates the ship at time T=0 """
        
        # Randomly select a cell on the interior of the ship (not on the edge)
        i = random.randint(1, (self.D-2)) 
        j = random.randint(1, (self.D-2))

        self.ship[i][j].open_cell()  # Open the selected cell
        
        flag = True
        while flag:
            blocked = []  # Keeps track of all blocked cells that neighbor exactly one open cell
            for i in range(self.D):
                for j in range(self.D):
                    if self.ship[i][j].get_state() == "#" and self.one_neighbor(i, j): 
                        blocked.append((i, j))  # Add cell to list if it is blocked and neighbors exactly one open cell
            if len(blocked) == 0:  # Exit the loop if there are no more such cells
                flag = False
                continue
            i, j = random.sample(blocked, 1)[0]  # Randomly select one of these cells
            self.ship[i][j].open_cell()  # Open the selected cell
        
        opened = []  # Keeps track of all open cells that neighbor exactly one open cell (dead ends)
        for i in range(self.D):
            for j in range(self.D):
                # neighboringCells = self.oneNeighbor(i, j)
                if self.ship[i][j].state == "O" and self.one_neighbor(i, j):
                    opened.append(self.closed_neighbors(i, j))
        
        opened = random.sample(opened, math.floor(len(opened)/2))  # Randomly select half of the dead ends
        for cell in opened:
            i, j = random.sample(cell, 1)[0]  # For each dead end, randomly select one closed neighbor
            self.ship[i][j].open_cell()  # Open the selected closed neighbor
        self.calculate_open_cells()

                    
        

    def distances_from_crew(self):
        """ Finds the distance from every cell to every other cell """
        start_cells = []
        for start_i in range(self.D):
            for start_j in range(self.D):
                if self.ship[start_i][start_j].is_open():
                    start_cells.append(self.ship[start_i][start_j])  # Adds all open cells in the ship to start_cells, since we have to calculate the distance to every other cell from every cell

        self.num_open_cells = len(start_cells)
        for i in range(0, len(start_cells)):  # For every open cell in the ship, calculate the distance to all other open cells
            fringe = []
            visited = []
            cur_state = (start_cells[i], 0) # The starting cell has a distance of 0
            fringe.append(cur_state)

            while not len(fringe) > 0: # Loop until all open cells in board visited

                cur_state = fringe.pop()
                cur_state[0].set_distance(start_cells[i].row, start_cells[i].col, cur_state[1]) 
                visited.append(cur_state[0])

                children = []
                cur_row = cur_state[0].row
                cur_col = cur_state[0].col
                if cur_row != 0:
                    children.append(self.ship[cur_row - 1][cur_col])
                if cur_row != (self.D - 1):
                    children.append(self.ship[cur_row + 1][cur_col])
                if cur_col != 0:
                    children.append(self.ship[cur_row][cur_col - 1])
                if cur_col != (self.D - 1):
                    children.append(self.ship[cur_row][cur_col + 1])

                for child in children:
                    if child.is_open() and (child not in visited) and (not child in fringe): # Prevent multiple of the same cell from entering the fringe
                        fringe.append((child, cur_state[1]+1))  # All children's distances get increased by 1 from parent distance


    def init_crew_prob_one(self):
        """ Initalizes the crew member probabilities for the case where there is only 1 crew member on the ship """
        p = 1 / (self.num_open_cells - 1)  # Every cell on the board besiddes the bot's starting cell has an equal probbaility of containing the crew member
        mask_func = lambda x: x.is_open() and not x.contains_bot()  
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship]) # returns all the open cells in the ship that do not contain the bot
        self.crew_probs[mask] = p
    
    def init_alien_prob_one(self):
        """ Initializes the alien probabailities for the case where thhere is only 1 alien on the ship """
        det_sq = self.get_sensor_region(self.bot.get_row(),self.bot.get_col())
        count = 0
        for row in det_sq:
            for elem in row:
                if elem.is_open():
                    count += 1  # Counts the number of open cells within the detection square
        num_open_out = self.num_open_cells - count  # Counts the number of open cells outside of the detection square
        p = 1 / num_open_out
        
        mask_func = lambda x: x.is_open()
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])

    
        x,y = self.get_det_sq_indicies()
        mask[x,y] = False

        self.alien_probs[mask] = p   # Sets all probabilities outside of the detection square to have a probability of containing an alien


    def get_det_sq_indicies(self):
        """ Returns the indices of the cells within the detection square """
        r1 = range(max(self.bot.row - self.k, 0), min(self.bot.row + self.k + 1, self.D))
        r2 = range(max(self.bot.col - self.k, 0),min(self.bot.col + self.k + 1, self.D))
        r3 = list(product(r1, r2))
        return [b[0] for b in r3], [b[1] for b in r3]


    # #Probability updates
    
    def get_out_det_sq_indicies(self):
        """ Returns indices of cells outside of the detection square """

        cent = (self.bot.row, self.bot.col)
        x = []
        y = []

        for i in range(self.D):
            for j in range(self.D):
                if not ((i in range(cent[0]-(self.k), cent[0]+(self.k)+1)) and (j in range(cent[1]-(self.k), cent[1]+(self.k)+1))):
                    x.append(i)
                    y.append(j)
        return x,y

    # One Alien, One Crew Probabilities 

    def one_one_alien_beep_update(self, beep):
        """ Adjusts alien probaabilities based on feedback from alien detection sensor (One alien on board) """
        #Beep is boolean, whether or not aliens were detected

        #indices within detection square based on current location of the bot
        x_in,y_in = self.get_det_sq_indicies()
        x_out, y_out = self.get_out_det_sq_indicies()

        if beep:
            #set probabilities outside the det sq to 0
            self.alien_probs[x_out, y_out] = 0
            
        else:
            #set probabilities inside the det sq to 0
            self.alien_probs[x_in,y_in] = 0
            
        alien_norm_factor = np.sum(self.get_alien_probs())
        self.alien_probs /= alien_norm_factor  # Normalization
        if(np.sum(self.get_alien_probs) == 0):
            self.init_alien_prob_one()

        

        np.around(self.alien_probs, decimals=10)  # For floating point integer errors
            
    
    def one_one_crew_beep_update(self, beep):
        """ Adjusts crew probabilities based on feedback from crew detection sensor (One crew member on board)"""
    
        if beep:
            #prob_function = lambda x: self.bot.get_beep_prob(x.row, x.col)
            prob_function = lambda x: self.bot.get_beep_prob(x.row, x.col)
            probs = np.asarray([list(map(prob_function, row)) for row in self.ship]) # probability off receiving a beep from any square

            
        else:
            prob_function = lambda x: 1 - self.bot.get_beep_prob(x.row, x.col)
            probs = np.asarray([list(map(prob_function, row)) for row in self.ship]) # probability of not receiving a beep from any square
    
        #Denominator
        sum_array = copy.deepcopy(self.crew_probs) 
        sum_array = np.multiply(sum_array, probs) 
        sum = np.sum(sum_array)
        #Numerator
        self.crew_probs = np.multiply(self.crew_probs, probs)

        #Fraction
        self.crew_probs /= sum

        #Avoids negative numbers from floating point error
        np.clip(self.crew_probs, a_min = 0,a_max =1, out = self.crew_probs)


    def one_one_bot_move_update(self):
        #This function applies when no alien or crew member was in the square we moved to (One crew member one bot on bord)

        bot_row = self.bot.get_row()
        bot_col = self.bot.get_col()

        #We know for sure no crew or alien in this square
        self.set_crew_probs(bot_row, bot_col, 0)
        self.set_alien_probs(bot_row, bot_col, 0)

        #Normalize the rest of the values

        crew_norm_factor = np.sum(self.crew_probs)
        alien_norm_factor = np.sum(self.alien_probs)

        self.crew_probs /= crew_norm_factor
        self.alien_probs /= alien_norm_factor
        np.around(self.alien_probs, decimals=10)


    def one_one_alien_move_update(self):
        #This function updates the alien probabilities when there is one alien on the board and the aliens move
        for i in range(self.D):
            for j in range(self.D):
                p = 0
                
                if self.ship[i][j].is_open():
                    #loop through adjacent cells
                    if (i-1) >= 0:
                        if self.ship[i-1,j].is_open():
                            p += ((self.alien_probs[i-1][j]) * (1/ self.open_neighbors[i-1][j]))
                    if (i +1) <= self.D -1:
                        if self.ship[i+1,j].is_open():
                            p += ((self.alien_probs[i+1][j]) * (1/ self.open_neighbors[i+1][j]))
                    if(j-1) >= 0:
                        if self.ship[i,j-1].is_open():
                            p += ((self.alien_probs[i][j-1]) * (1/ self.open_neighbors[i][j-1]))
                    if(j+1) <= self.D-1:
                        if self.ship[i,j+1].is_open():
                            p += ((self.alien_probs[i][j+1]) * (1/ self.open_neighbors[i][j+1]))
                    self.set_alien_probs(i,j,p)
                    np.around(self.alien_probs, decimals=10)

    def open_cell_indices(self):
        # Returns the combination of all open cell pairs with or without replacement
        mask_func = lambda x: x.is_open()
        mask = np.asarray([list(map(mask_func, row)) for row in self.ship])
        #mask_trues = np.where(mask==True)
        return mask
        

