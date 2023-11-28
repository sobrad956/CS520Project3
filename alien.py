import numpy as np
import random


class Alien:

    def __init__(self, row, col, ship):
        self.row = row
        self.col = col
        self.ship = ship
        self.ship.ship[self.row][self.col].add_alien()

    def move_up(self):
        self.ship.ship[self.row][self.col].remove_alien()
        self.row -= 1
        self.ship.ship[self.row][self.col].add_alien()
        if self.ship.ship[self.row][self.col].contains_bot():
            print("Mission Failed, Bot Captured")
            return True
        return False

    def move_down(self):
        self.ship.ship[self.row][self.col].remove_alien()
        self.row += 1
        self.ship.ship[self.row][self.col].add_alien()
        if self.ship.ship[self.row][self.col].contains_bot():
            print("Mission Failed, Bot Captured")
            return True
        return False
    
    def move_right(self):
        self.ship.ship[self.row][self.col].remove_alien()
        self.col += 1
        self.ship.ship[self.row][self.col].add_alien()
        if self.ship.ship[self.row][self.col].contains_bot():
            print("Mission Failed, Bot Captured")
            return True
        return False

    def move_left(self):
        self.ship.ship[self.row][self.col].remove_alien()
        self.col -= 1
        self.ship.ship[self.row][self.col].add_alien()
        if self.ship.ship[self.row][self.col].contains_bot():
            print("Mission Failed, Bot Captured")
            return True
        return False
    
    def choose_action(self):
        avail_choices = [0]
        if self.col != 0 and self.ship.ship[self.row][self.col - 1].is_open():
            avail_choices.append(1)
        if self.col != self.ship.D-1 and self.ship.ship[self.row][self.col + 1].is_open():
            avail_choices.append(2)
        if self.row != 0 and self.ship.ship[self.row - 1][self.col].is_open():
            avail_choices.append(3)
        if self.row != self.ship.D-1 and self.ship.ship[self.row + 1][self.col].is_open():
            avail_choices.append(4)
        return random.choice(avail_choices)
    
    def move(self):
        choice = self.choose_action()
        if choice == 1:
            return self.move_left()
        if choice == 2:
            return self.move_right()
        if choice == 3:
            return self.move_up()
        if choice == 4:
            return self.move_down()