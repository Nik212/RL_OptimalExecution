import numpy as np
from math import isclose
class TradingSimulator:

    def __init__(self, datahandler, steps):
        self.steps = steps #N+1
        self.datahandler = datahandler
        self.X_0 = 100000
        self.q = 5000
        self.rho = 2.2231
        self.lambda_ = 1/(2*self.q)
        self.spread = 0.05
        self.kappa = 1/self.q - self.lambda_
        self.taus = [n*1/self.steps for n in range(0,self.steps)] #T

        self.step = 0
        self.prices = np.zeros(self.steps)
        self.actions = np.zeros(self.steps)
        self.strategy_costs = np.ones(self.steps)
        self.inventory = self.X_0

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.prices.fill(0)
        self.strategy_costs.fill(0)
        self.inventory = self.X_0

        return np.array([np.log(self.datahandler.data[0]), self.inventory, self.steps])
    
    def constraint_(self, action):
        C = 100000
        if action <= 0:
            return (abs(action)+1)*C
        #if self.step == self.steps-1 and not isclose(np.sum(self.actions),self.X_0):
        #    return (abs(action)+1)*C
        #if self.step == self.steps-1 and isclose(np.sum(self.actions),self.X_0):
        #    return 0
        return action

    def take_step(self, action):
        actions_percent = action/self.inventory
        if self.step == self.steps-1 and isclose(np.sum(self.actions[:self.steps - 1]),self.X_0):
            
            action = 0

            self.actions[self.step] = action
            self.inventory -= action
            
            self.prices[self.step] = self.datahandler.data[self.step]

            reward = 0

            self.strategy_costs[self.step] = reward
            state = np.array([
                    self.prices[self.step] - self.prices[self.step-1],
                    self.inventory,
                    max(abs(self.steps - self.step), 0)
                ])
            info = self.prices
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done
        
        if self.step == self.steps-1 and (not isclose(np.sum(self.actions[:self.steps - 1]),self.X_0)):
            C = 10000            
            action = (abs(action)+1)*C

            self.actions[self.step] = 0
            self.inventory -= 0
            
            self.prices[self.step] = self.datahandler.data[self.step]

            reward = -((self.prices[self.step]+action/(2*self.q)))*(abs(action)+1)*C

            self.strategy_costs[self.step] = reward

            state = np.array([
                    self.prices[self.step] - self.prices[self.step-1],
                    self.inventory,
                    max(abs(self.steps - self.step), 0)
                ])
            info = self.prices
            self.step += 1
            if self.step >= self.steps:
                done = True
            else:
                done = False
            return reward, state, info, done

        self.actions[self.step] = action
        self.inventory -= action

        sum_ = np.sum([self.actions[i]*self.kappa*np.exp(-self.rho*self.taus[self.step]*(self.step - i)) for i in range(self.step)])
        self.prices[self.step] = self.datahandler.data[self.step] + self.lambda_*np.sum(self.actions[:self.step]) + self.spread/2 + sum_

        reward = -((self.prices[self.step]+action/(2*self.q)))*self.constraint_(action)
        self.strategy_costs[self.step] = reward
        
        if self.step == 0:
             state = np.array([
                    self.prices[self.step] - self.datahandler.data[0],
                    self.inventory,
                    max(abs(self.steps - self.step), 0)
                ])
        else:
            state = np.array([
                    self.prices[self.step] - self.prices[self.step-1],
                    self.inventory,
                    max(abs(self.steps - self.step), 0)
                ])
            
        self.step += 1
        info = self.prices
        if self.step >= self.steps:
            done = True
        else:
            done = False
        return reward, state, info, done