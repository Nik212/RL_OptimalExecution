import numpy as np

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
        self.taus = [n*1/self.steps for n in range(0,self.steps+1)] #T

        self.step = 0
        self.prices = np.zeros(self.steps)
        self.actions = np.zeros(self.steps)
        self.strategy_costs = np.ones(self.steps)
        self.inventory = self.X_0

        print('Environment set-up:')
        print('Trading period:', self.steps)

        

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.prices.fill(0)
        self.strategy_costs.fill(0)
        self.inventory = self.X_0
        print('Initial inventory', self.inventory)

    def take_step(self, action):
        self.actions[self.step] = action
        print('Current step', self.step)
        if self.step != 0:
            self.inventory -= action
            print('Inventory after trade:', self.inventory)

        if self.step == 0:
            self.prices[self.step] = self.datahandler.data[self.step]
        else:
            sum_ = np.sum([self.actions[i]*self.kappa*np.exp(-self.rho*self.taus[self.step]*(self.step - i)) for i in range(self.step)])
            self.prices[self.step] = self.datahandler.data[self.step] + self.lambda_*(self.inventory) + self.spread/2 + sum_
        print('Current market price:', self.prices[self.step])

        reward = (self.prices[self.step]+action/(2*self.q))*action
        self.strategy_costs[self.step] = reward
        
        
        
        if self.step == 0:
            state = (self.prices[self.step], self.inventory, self.steps)
        else:
            state = (
                self.prices[self.step] - self.prices[self.step-1],
                self.inventory,
                max(abs(self.steps - self.step), 0)
            )


        self.step += 1
        info = self.prices[self.step]
        return reward, state, info