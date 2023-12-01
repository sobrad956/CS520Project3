import math
import numpy as np
import random


class Bot:
    """The bot class stores relevant information about the bot"""

    def __init__(self, row, col, k, ship, type, alpha):
        self.row = row
        self.col = col
        self.k = k
        self.ship = ship
        self.type = type
        self.ship.ship[self.row][self.col].add_bot()
        self.ship.set_bot_loc(self.row, self.col)
        self.alpha = alpha

    def move_up(self):
        """Bot moves up"""
        self.ship.ship[self.row][self.col].remove_bot()
        self.row -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_down(self):
        """Bot moves down"""
        self.ship.ship[self.row][self.col].remove_bot()
        self.row += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_right(self):
        """Bot moves right"""
        self.ship.ship[self.row][self.col].remove_bot()
        self.col += 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def move_left(self):
        """Bot moves left"""
        self.ship.ship[self.row][self.col].remove_bot()
        self.col -= 1
        self.ship.ship[self.row][self.col].add_bot()
        if self.ship.ship[self.row][self.col].contains_alien():
            return ("Mission Failed, Bot Captured")

    def found_crew(self):
        """Checks if a crew member is rescued"""
        if self.ship.ship[self.row][self.col].contains_crew():
            self.ship.ship[self.row][self.col].remove_crew()
            return True

    """Return the location of the bot"""
    def get_row(self):
        return self.row

    def get_col(self):
        return self.col

    
    def get_type(self):
        """Check bot number"""
        return self.type

    def get_sensor_region(self, i, j):
        """ Returns the cells within the alien sensor region """
        return self.ship.get_sensor_region(self.row, self.col)

    def detect_alien(self):
        """ Returns True if an alien is in the sensor region, False otherwise """
        region = self.get_sensor_region(self.row, self.col)
        for r in range(len(region)):
            for c in range(len(region[0])):
                if region[r][c].contains_alien():
                    return True
        return False

    def get_beep_prob(self, row, col):
        #crewnum is the crew number's index
        d = self.ship.ship[row][col].distances[self.row][self.col]
        prob = math.exp(-self.alpha * (d - 1))
        #print(prob)
        return prob
    

    def detect_crew(self, crew_locs):
        """ Returns a beep with probability for each crew member based on distance """
        for i in range(len(crew_locs)):
            prob = self.get_beep_prob(crew_locs[i].row, crew_locs[i].col)
            if random.random() < prob:
                return True
        return False

    
    def find_mult_max(self, values):
        """Returns the indicies of all maximum values if more than one exist"""
        mult_max = []
        idx = 0
        max_val = max(values)
        for val in values:
            if val == max_val:
                mult_max.append(idx)
            idx += 1
        return mult_max

    def find_mult_min(self, values):
        #Same as above but for min values
        mult_min = []
        idx = 0
        min_val = min(values)
        for val in values:
            if val == min_val:
                mult_min.append(idx)
            idx += 1
        return mult_max

    def bot1_move(self):
        """Movement functionality for bot1"""

        next_move = 10
        cur_row = self.row
        cur_col = self.col

        """These if else statements check all adjacent squares to see if they are open and 
        if they are it stores the probability that each contains an alien and a crew"""
        if cur_row > 0 and self.ship.ship[cur_row-1][cur_col].is_open():
            up_crew_prob = self.ship.get_crew_probs()[cur_row-1][cur_col]
            up_alien_prob = self.ship.get_alien_probs()[cur_row-1][cur_col]
        else:
            up_crew_prob = -1
            up_alien_prob = 100
        
        if cur_row < self.ship.D - 1 and self.ship.ship[cur_row+1][cur_col].is_open():
            down_crew_prob = self.ship.get_crew_probs()[cur_row+1][cur_col]
            down_alien_prob = self.ship.get_alien_probs()[cur_row+1][cur_col]
        else:
            down_crew_prob = -1
            down_alien_prob = 100

        if cur_col > 0 and self.ship.ship[cur_row][cur_col-1].is_open():
            left_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col-1]
            left_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col-1]
        else:
            left_crew_prob = -1
            left_alien_prob = 100
        
        if cur_col < self.ship.D-1 and self.ship.ship[cur_row][cur_col+1].is_open():
            right_crew_prob = self.ship.get_crew_probs()[cur_row][cur_col+1]
            right_alien_prob = self.ship.get_alien_probs()[cur_row][cur_col+1]
        else:
            right_crew_prob = -1
            right_alien_prob = 100

        crew_probs = [left_crew_prob, right_crew_prob, up_crew_prob, down_crew_prob]
        alien_probs = [left_alien_prob, right_alien_prob, up_alien_prob, down_alien_prob]

        
        #Find the indicies of the max crew probabilities and chooses one at random
        mult_max = self.find_mult_max(crew_probs)
        max_idx = random.choice(mult_max)
        
        while(next_move == 10):
            #If there is no guaranteed safe move, the bot stays still
            if 0 not in alien_probs:
                next_move = 100
            
            #If the max crew prob neighbor definitely doesn't contain an alien that move is selected
            elif alien_probs[max_idx] == 0:
                next_move = max_idx
            #If the max crew neighbor might contain an alien, we choose another
            else:
                crew_probs[max_idx] = -1
                mult_max = self.find_mult_max(crew_probs)
                max_idx = random.choice(mult_max)
        if next_move == 0:
            self.move_left()
            return 0
        elif next_move == 1:
            self.move_right()
            return 1
        elif next_move == 2:
            self.move_up()
            return 2
        elif next_move == 3:
            self.move_down()
            return 3
        else:
            return 4

    
