import random
import pickle
import pandas as pd
import os


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.best_reward = -float('inf')
        self.best_filename = "best_policy.pkl"

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.
        try:
            with open(filename, 'rb') as p:
                self.q = pickle.load(p)
                print(f"Loaded file: {filename}.pkl")
        except FileNotFoundError:
            print(f"Could not find {filename}.pkl; initialize Q-Values from scratch")

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.

        pkl_file = f"{filename}.pkl"
        with open(pkl_file, 'wb') as p:
            pickle.dump(self.q, p)
        print(f"Wrote to file: {pkl_file}")
        
        q_list = [(state, action, q_value) for (state, action), q_value in self.q.items()]
        df = pd.DataFrame(q_list, columns=["State", "Action", "Q-Value"])
    
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Wrote Q-values to CSV file: {csv_file}")

    def saveBestPolicy(self, reward):
        '''
        Save the current Q-values as the best policy if the current reward is the highest
        '''
        if reward > self.best_reward:
            print(f"New best reward: {reward}. Saving policy.")
            self.best_reward = reward
            self.saveQ(self.best_filename)  # Save the Q-table for the best policy
        else:
            print(f"Current reward: {reward} does not exceed best reward: {self.best_reward}. Policy not saved.")

    def loadBestPolicy(self):
        '''
        Load the best policy (Q-values) from a saved file.
        '''
        if os.path.exists(self.best_filename):
            self.loadQ(self.best_filename)
            print(f"Loaded best policy from {self.best_filename}")
        else:
            print("No best policy found. Starting from scratch.")
        


    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        if random.uniform(0,1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            max_q = max(q_values)

            max_actions = []

            for action, qval in zip(self.actions, q_values):
                if qval == max_q:
                    max_actions.append(action)

            action = random.choice(max_actions) #randomly pick from the best actions 

        if return_q:
            return action, max_q
        else:
            return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        current_q = self.getQ(state1, action1)
        
        next_state_max_q = max([self.getQ(state2,action) for action in self.actions])

        updated_q = current_q + self.alpha * (reward + self.gamma * next_state_max_q - current_q)

        self.q[state1,action1] = updated_q