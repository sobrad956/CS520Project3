import numpy as np
import random
import math
import matplotlib.pyplot as plt

from ship import Ship
from alien import Alien
from bot import Bot


def experiment1(k, alphas, boards):
    """This runs the 1 alien, 1 crew member experiments (bots 1 and 2)"""
    numBoards = len(boards)
    numTrials = 50
    bots = [1]
    avg_crew_saved = np.zeros((2, len(alphas)))
    avg_moves_to_save = np.zeros((2, len(alphas)))
    prob_success = np.zeros((2, len(alphas)))
    
    board_count = 0

    for board in boards:
        shp = board
        for a, alpha in enumerate(alphas):
            for botnum in bots:
                for trial in range(numTrials):
                    i, j = shp.get_unoccupied_cell()
                    bot = Bot(i, j, k, shp, botnum, alpha)
                    shp.bot = bot

                    start_cells = []
                    i, j = shp.get_unoccupied_cell()
                    shp.ship[i][j].add_crew()
                    start_cells.append(shp.ship[i][j])

                    i, j = shp.get_unoccupied_alien_cell(k)
                    alien = Alien(i, j, shp)

                    shp.print_ship()
                    shp.distances_from_crew()
                    #print("Calculated distances")
                    shp.init_crew_prob_one()
                    # print("init crew prob: ")
                    # print(shp.get_crew_probs())
                    print()
                    
                    shp.init_alien_prob_one()
                    # print("init alien prob: ")
                    # print(shp.get_alien_probs())
                    print()

                    shp.print_ship()
                    print('Experiment: Experiment 1', 'ExpBoard:', board_count,'Alpha:', alpha, ' Botnum:', botnum, ' Trial:', trial)
                    T = 0
                    flag = True
                    while flag:
                        #if T > 40:
                        #    break
                        if botnum == 1:
                            bot.bot1_move()
                        else:
                            bot.bot2_move()

                        i = bot.row
                        j = bot.col

                        if shp.ship[i][j].contains_alien():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
                            flag = False
                            break
                        if shp.ship[i][j].contains_crew():
                            print(f"Saved: {T}")
                            avg_crew_saved[botnum - 1][a] += 1 / (numBoards * numTrials)
                            prob_success[botnum - 1][a] += 1 / (numBoards * numTrials)
                            shp.ship[i][j].remove_crew()
                            flag = False
                            break
                        shp.one_one_bot_move_update()
                        #print("bot move: ", shp.get_crew_probs())
                        if alien.move():
                            print(f"Dead: {T}")
                            avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
                            flag = False
                            break
                        shp.one_one_alien_move_update()
                        
                        alien_beep = bot.detect_alien()
                        shp.one_one_alien_beep_update(alien_beep)
                        crew_beep = bot.detect_crew(start_cells)
                        shp.one_one_crew_beep_update(crew_beep)
                        
                        #print("crew beep: ", shp.get_crew_probs())
                        #print(shp.get_alien_probs())
                        #print()

                        T += 1
                    shp.empty_ship()
        board_count += 1  

    alphas = [str(x) for x in alphas]
    
    plt.plot(alphas, avg_moves_to_save[0], label='Bot 1')
    plt.plot(alphas, avg_moves_to_save[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Average Number of Moves Needed to Rescue all Crew Members')
    plt.title('Average Number of Moves Needed to Rescue all Crew Members (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1/experiment1_plot1.png')
    np.save('experiment1/experiment1_plot1.npy', avg_moves_to_save)
    plt.plot()
    plt.close()

    plt.plot(alphas, avg_crew_saved[0], label='Bot 1')
    plt.plot(alphas, avg_crew_saved[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Average Number of Crew Members Saved')
    plt.title('Average Number of Crew Members Saved (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1/experiment1_plot2.png')
    np.save('experiment1/experiment1_plot2.npy', avg_crew_saved)
    plt.plot()
    plt.close()

    plt.plot(alphas, prob_success[0], label='Bot 1')
    plt.plot(alphas, prob_success[1], label='Bot 2')
    plt.xlabel('Value for alpha')
    plt.ylabel('Probability of Successfully Saving all Crew Members')
    plt.title('Probability of Successfully Saving all Crew Members (One Alien, One Crew) vs Alpha')
    plt.legend(loc='best')
    plt.savefig('experiment1/experiment1_plot3.png')
    np.save('experiment1/experiment1_plot3.npy', prob_success)
    plt.plot()
    plt.close()
    





def main(k, D):
    desired_width = 7*7*2
    np.set_printoptions(linewidth=desired_width)
    crewnum = 2
    shp = Ship(k, D)
    shp.generate_ship()

    i, j = shp.get_unoccupied_cell()
    bot = Bot(i, j, k, shp, 1, 0.1)
    shp.bot = bot

    start_cells = []
    for num in range(crewnum):
        i, j = shp.get_unoccupied_cell()
        shp.ship[i][j].add_crew()
        start_cells.append(shp.ship[i][j])

    aliens = []
    for o in range(2):
        i, j = shp.get_unoccupied_alien_cell(k)
        alien = Alien(i, j, shp)
        aliens.append(alien)

    print("INIT")
    shp.distances_from_crew()
    shp.init_crew_prob_two()
    #print(shp.get_crew_probs())
    print()
    shp.init_alien_prob_two()
    #print(shp.get_alien_probs())
    print()
    shp.print_ship()
    print()
    print("BOT MOVES")
    bot.bot1_move()
    shp.print_ship()
    print()
    shp.one_two_bot_move_update()
    #print(shp.get_crew_probs())
    print()
    print(shp.get_alien_probs())
    print()
    print("ALIEN MOVES")
    for al in aliens:
        al.move()
    shp.print_ship()
    print()
    shp.one_one_alien_move_update()
    #print(shp.get_crew_probs())
    print()
    print(shp.get_alien_probs())
    print()
    print("ALIEN BEEP UPDATE")
    alien_beep = bot.detect_alien()
    print(alien_beep)
    shp.one_one_alien_beep_update(alien_beep)
    #print(shp.get_crew_probs())
    print()
    print(shp.get_alien_probs())
    print()
    print("CREW BEEP UPDATE")
    crew_beep = bot.detect_crew(start_cells)
    print(crew_beep)
    shp.one_two_crew_beep_update(crew_beep)
    #print(shp.get_crew_probs())
    print()
    print(shp.get_alien_probs())
    print()





    #print(bot.detect_alien())
    #print(bot.detect_crew(start_cells))
    #beep = bot.detect_crew(start_cells)
    #shp.one_one_crew_beep_update(beep)



#     numBoards = 30
#     numTrials = 10
#     bots = [1, 2]
#     avg_crew_saved = np.zeros((2, len(alphas)))
#     avg_moves_to_save = np.zeros((2, len(alphas)))
#     prob_success = np.zeros((2, len(alphas)))
#
#     shp = Ship()
#     for board in range(numBoards):
#         shp.empty_ship()
#         shp.generate_ship()
#
#         for a, alpha in enumerate(alphas):
#             for botnum in bots:
#                 for trial in range(numTrials):
#                     i, j = shp.get_unoccupied_cell(False)
#                     bot = Bot(i, j, k, shp, botnum)
#
#                     i, j = shp.get_unoccupied_cell(True)
#                     shp.ship[i][j].add_crew()
#                     shp.set_crew_loc(i, j)
#
#                     i, j = shp.get_unoccupied_alien_cell(k)
#                     alien = Alien(i, j, shp)
#
#                     #Initialize ship probabilities
#
#                     shp.print_ship()
#                     print('Board:', board, ' Botnum:', botnum, ' Trial:', trial)
#                     T = 0
#                     flag = True
#                     while flag:
#                         #SENSORS
#                         #aliendetected = bot.detect_alien()
#                         #crewbeep = bot.detect_crew(1)
#                         #Update probabilities
#                         #BOT MOVES
#                         #i, j = bot.move() /need logic to pick highest prob square from ship.crew_probs
#                         if shp.ship[i][j].contains_alien():
#                             print(f"Dead: {T}")
#                             avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
#                             flag = False
#                             break
#                         if shp.ship[i][j].contains_crew():
#                             print(f"Saved: {T}")
#                             avg_crew_saved[botnum - 1][a] += 1 / (numBoards * numTrials)
#                             prob_success[botnum - 1][a] += 1 / (numBoards * numTrials)
#                             shp.ship[i][j].remove_crew()
#                             flag = False
#                             break
#                         # alien moves
#                         # alien.move()
#                         if shp.ship[i][j].contains_bot():
#                             print(f"Dead: {T}")
#                             avg_moves_to_save[botnum - 1][a] += T / (numBoards * numTrials)
#                             flag = False
#                             break
#                         T += 1
#                     shp.empty_ship()


if __name__ == "__main__":
    boards = []
    print("top of main")
    for i in range(1):
        #ship takes in k, D
        shp = Ship(1, 25)
        shp.generate_ship()
        print("ship generated")
        boards.append(shp)
        
    #main(1, 5)

    #experiement takes k, alpha, boards

    experiment1(3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], boards)
    