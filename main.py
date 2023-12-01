import numpy as np
import random
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot


def simulateData(k,boards):
    """This runs the 1 alien, 1 crew member experiments (bots 1 and 2)"""
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
    


if __name__ == "__main__":
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
    #experiment1(k, boards)
    simulateData(k, boards)
    