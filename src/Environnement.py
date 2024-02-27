import numpy as np
import matplotlib.pyplot as plt
import random as rng

class State:
    def __init__(self, name, type=0):
        #initialise the state
        self.number = name
        self.type = type
        self.rewards = [-0.04, 1, -1, float('nan')][int(type)]
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.neighbors = {}
        self.policy = None
        #variables for the value iteration
        self.utility = 0
        #variables for the Q-learning
        self.q_table = { action: 0 for action in self.actions}
        self.Nsa_table = { action: 0 for action in self.actions}
        #to indicate the start state
        if self.number == (0, 0):
            self.start = True
        else:
            self.start = False
    #method to add a neighbor to the state 
    def add_neighbor(self, action, list_state_proba):
        self.neighbors[action] = list_state_proba
        

class Environment:
    #class to represent the environment (grid, method to play, value iteration, Q-learning, etc.)
    def __init__(self, grid, gamma=0.99, threshold=0.01, episode=200, alpha=0.1):
        #initialise the environment
        self.grid = grid #grid of the environment
        self.states = [] #list of states
        self.gamma = gamma
        self.episode = episode
        self.alpha = alpha
        self.threshold = threshold
        self.trace = {}
        self.init_states()

    def init_states(self):
        """
        Populate the states list with the states of the grid and their neighbors and transitionnal model
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.states.append(State((i, j), self.grid[i][j]))
        for state in self.states:
            if state.type == 0:
                self.init_transition(state)
                
    def is_a_wall(self, number):
        """
        Check if the state is a wall
        """
        i, j = number
        if i < 0 or j < 0 or i >= len(self.grid) or j >= len(self.grid[i]):
            return True
        return self.grid[i][j] == 3
           
    def init_transition(self, state):
        """
        Create the transitionnal model for the state with the probabilities of the neighbors given an action
        """
        i, j = state.number
        directions = {'up': (1, 0), 'down': (-1, 0), 'left': (0, -1), 'right': (0, 1)} #directions for the actions
        fallbacks = {'up': ['left', 'right'], 'down': ['left', 'right'], 'left': ['up', 'down'], 'right': ['up', 'down']} #possible end states

        for action in state.actions:

            proba_current = 0
            
            #compute possible states coordinates
            current_state = (i, j)
            next_state_wanted = (i + directions[action][0], j + directions[action][1])
            next_state_fallback1 = (i + directions[fallbacks[action][0]][0], j + directions[fallbacks[action][0]][1])
            next_state_fallback2 = (i + directions[fallbacks[action][1]][0], j + directions[fallbacks[action][1]][1])
            
            #init the list of probabilities
            list_state_proba = []
            
            #check if the desired state is a wall
            if self.is_a_wall(next_state_wanted):
                proba_current = 0.8 #if it's a wall, the probability of staying in the current state is 0.8
            else:
                proba_desired = 0.8 #if not, the probability of going to the desired state is 0.8
                list_state_proba.append((next_state_wanted, proba_desired)) #add the state-probability 
                
            #check if the fallback states are walls and update the probabilities accordingly
            if self.is_a_wall(next_state_fallback1):
                proba_current+=0.1  #if it's a wall, the probability of staying in the current state increase by 0.1
            else:
                proba_fallback1 = 0.1 
                #if not a wall, add new neighbor
                list_state_proba.append((next_state_fallback1, proba_fallback1))
            if self.is_a_wall(next_state_fallback2):
                proba_current+=0.1
            else:
                proba_fallback2 = 0.1
                #if not a wall, add new neighbor
                list_state_proba.append((next_state_fallback2, proba_fallback2))
                
            #add the current or desired state to the neighbors  
            if proba_current != 0:
                list_state_proba.append((current_state, proba_current))
            state.add_neighbor(action, list_state_proba)    
            
    def get_transitionnal_model(self):
        """
        Print the transitionnal model of the environment
        """
        for state in self.states:
            print("====", state.number, "====")
            if self.is_a_wall(state.number):
                print("WALL")
                continue
            if state.type != 0:
                print("TERMINAL")
                continue
            for action in state.actions:
                print(action + " : ")
                for neighbor in state.neighbors[action]:
                    print(neighbor)
                    
    def play(self, state, action):
        """
        Give the next state given the current state and the action using the transitionnal model
        """
        next_state = rng.choices([neighbor[0] for neighbor in state.neighbors[action]], [neighbor[1] for neighbor in state.neighbors[action]])[0]
        return next_state
        
                    
    def get_state(self, number):
        """
        Getter function that returns the state given its coordinates
        """
        for state in self.states:
            if state.number == number:
                return state
        return None
    
    def plot_grid(self):
        """
        function to visualize the environment
        """
        color_map = {0: 'white', 1: 'green', 2: 'red', 3: 'black', 4: 'yellow'}
        
        grid = self.grid
        for state in self.states:
            if state.start:
                grid[state.number] = 4
        fig, ax = plt.subplots()
        for (i,j), value in np.ndenumerate(grid):
            ax.add_patch(plt.Rectangle((j,i), 1, 1, fill=True, color=color_map[value]))
            ax.text(j+0.5, i+0.5, self.get_state((i,j)).rewards  , ha='center', va='center')
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(0, grid.shape[0])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()

        
    def calculate_utility(self, state, step=0, number=(0,0)):
        """
        Calculate the utility of the state using the Bellman equation
        """
        bes_Q = -float('inf') #initialise the best Q value
        best_action = None 
        for action in state.actions:
            Q = 0
            for neighbor in state.neighbors[action]:
                next_state = self.get_state(neighbor[0])
                Q += neighbor[1] * (next_state.rewards + self.gamma * next_state.utility)
            self.trace[step][number]["Q_"+action] = Q
            if Q > bes_Q:
                bes_Q = Q
                best_action = action
        state.policy = best_action
        return bes_Q
    
    def value_iteration(self):
        """
        Compute the utility of the states using the value iteration algorithm
        """
        step=0
        delta = float('inf')
        self.trace[step] = {state.number: state.utility for state in self.states if state.type != 3}
        self.trace[step]["delta"] = "inf"
        while delta > self.threshold:
            delta = 0
            step+=1
            self.trace[step] = {}
            for state in self.states:
                if state.type != 0:
                    continue
                U = state.utility
                self.trace[step][state.number]={}
                state.utility = self.calculate_utility(state, step, state.number)
                self.trace[step][state.number]['U'] = state.utility
                delta+= abs(U - state.utility)
            self.trace[step]["delta"] = delta
        self.save_trace_value("../log/log-file_VI.txt")

    def get_policy(self):
        """
        Print the policy of the states
        """
        for state in self.states:
            if state.type != 0:
                continue
            print(state.number, state.policy)
            
    def save_trace_value(self, filename):
        """
        Function to save the trace of the value iteration algorithm into a text file
        """
        texte = []
        texte.append(f"MDP à {len(self.states)} états :\n")
        state_numbers = [(str(state.number)+", ") for state in self.states]
        texte.append(' '.join(state_numbers) + '\n')
        texte.append(f"Etats terminaux : {[state.number for state in self.states if state.type == 1 or state.type == 2] }\n")
        texte.append(f"But: {[state.number for state in self.states if state.type == 1] }\n")
        texte.append(f"Gammma : {self.gamma}\n")
        texte.append(f"Seuil : {self.threshold}\n")
        texte.append("====================================================================================================\n")
        texte.append("\t\tTrace de la valeur :\n")
        texte.append("====================================================================================================\n")
        for step in range(len(self.trace)):
            texte.append(f"Etape {step} : \n")
            if step == 0: 
                texte.append("On initialise tous les U à 0\n")
                texte.append("====================================================================================================\n")
                continue
            for state_number, value in self.trace[step].items():
                if state_number == "delta":
                    continue
                state = self.get_state(state_number)
                number = state.number[0]*len(self.grid[0])+state.number[1]
                line="U'_"+str(number)+" = max{ "
                for action in state.actions:
                    part_text=""
                    part_number=""
                    for neighbor, proba in state.neighbors[action]:
                        part_text+=f"{proba}x[{state.rewards}+{self.gamma}xU_{neighbor[0]*len(self.grid[0])+neighbor[1]}] + "
                        part_number+=f"{self.trace[step][state_number]['Q_'+action]:.5g}, "
                    part_text = part_text[:-2]
                    line+=f"{part_text}, "
                line = line[:-2]
                line+=f" {'}'} = max{'{'} {part_number[:-2]} {'}'} = {self.trace[step][state_number]['U']:.5g}\n"
                texte.append(line)
            if step < len(self.trace)-1:   
                texte.append(f"Delta = {self.trace[step]['delta']} > Seuil = {self.threshold}\n")
            else:
                texte.append(f"Delta = {self.trace[step]['delta']} < Seuil = {self.threshold}\n")
                texte.append("On a atteint le seuil, on met maintenant à jour les politiques\n")
            texte.append("====================================================================================================\n")
        texte.append("\n\n\n")
        texte.append("\t\tPolitiques optimales :\n")
        texte.append("====================================================================================================\n")
        for state in self.states:
            if state.type != 0:
                continue
            texte.append(f"π({state.number}) = argmax_a Q(s,a) = {state.policy}\n")
        
        with open(filename, "w") as file:
            file.writelines(texte)
            
    def q_learning(self):
        """
        Compute the policy of each states using the Q-learning algorithm
        """
        step = 0
        trace=[]
        for episode in range(self.episode):
            state = self.get_state((0,0))
            trace.append("====================================\n")
            trace.append(f"Episode {episode} : \n")
            trace.append("====================================\n")
            #while the state is not an end state
            while state.type == 0:
                trace.append("------------------------------------\n")
                trace.append(f"Step {step} : \n")
                step+=1
                #choose the action with the highest Q value
                action = state.actions[np.argmax(list(state.q_table.values()))]
                trace.append(f"State : {state.number}\n")
                trace.append(f"Action : {action}\n")
                #maj N(s,a)
                #update the number of times the action has been chosen
                state.Nsa_table[action]+=1
                trace.append(f"N(s,a) : {state.Nsa_table}\n")
                #simulate the action and get the next state
                next_state_number = self.play(state, action)
                trace.append(f"Next state : {next_state_number}\n")
                next_state = self.get_state(next_state_number)
                #save the reward
                reward = next_state.rewards
                trace.append(f"Reward : {reward}\n")
                #update the Q value 
                state.q_table[action] += self.alpha * (1/state.Nsa_table[action]) * (reward + self.gamma * max(next_state.q_table.values()) - state.q_table[action])
                trace.append(f"Q({state.number},{action}) = Q({state.number},{action}) + α/N(s,a) x (R + γxmax(Q(s',a')) - Q(s,a))\n")
                trace.append(f"Q({state.number},{action}) = Q({state.number},{action}) + {self.alpha}x{1/state.Nsa_table[action]}x({reward} + {self.gamma}x{max(next_state.q_table.values())} - {state.q_table[action]}) = {state.q_table[action]}\n")
                state = next_state
        trace.append("====================================\n")
        trace.append("Update Politic et utility\n")
        trace.append("====================================\n")
        #update the policy and the utility of the states
        for state in self.states:
            #if the state is not an end state
            if state.type == 0:
                #update the policy with the action with the highest Q value
                state.policy = state.actions[np.argmax(list(state.q_table.values()))]
                trace.append(f"π({state.number}) = argmax_a Q(s,a) = {state.policy}\n")
                #update the utility with the highest Q value
                state.utility = max(state.q_table.values())
                trace.append(f"U({state.number}) = max_a Q(s,a) = {state.utility}\n")
            #if the state is an end state
            elif state.type == 1 or state.type == 2:
                trace.append(f"{state.number} is an End State\n")
            #if the state is a wall
            else:
                trace.append(f"{state.number} is a Wall\n")
            trace.append("------------------------------------\n")
        self.save_trace_ql("../log/log-file_QL.txt", trace)
    
    def save_trace_ql(self, filename, trace):
        """
        Function to save the trace of the Q-learning algorithm into a text file
        """
        texte = []
        texte.append(f"MDP à {len(self.states)} états :\n")
        state_numbers = [(str(state.number)+", ") for state in self.states]
        texte.append(' '.join(state_numbers) + '\n')
        texte.append(f"Etats terminaux : {[state.number for state in self.states if state.type == 1 or state.type == 2] }\n")
        texte.append(f"But: {[state.number for state in self.states if state.type == 1] }\n")
        texte.append(f"Gammma : {self.gamma}\n")
        texte.append(f"Nombre d'épisodes : {self.episode}\n")
        texte.append(f"Alpha : {self.alpha}\n")
        texte.append("====================================================================================================\n")
        texte.append("\t\tQ-Learning :\n")
        texte.append("====================================================================================================\n")
        for line in trace:
            texte.append(line)          
        with open(filename , "w") as file:
            file.writelines(texte)