import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, data, trading_period):
        self.data = pd.Series(data)
        self.step = 0
        self.offset = None
        self.trading_period = trading_period
    
    def process_data_(self):        
        return self.data
    
    def get_data_(self):
        return self.data
    
    def reset(self):
        """
        Provides starting index for time series and resets step
        
        """

        high = len(self.data.index) - self.trading_period 
        self.offset = np.random.randint(low=0, high=1)
        self.step = 0
        print('Starting date:', str(self.data.index[0]))
        print('Starting fundamental price:',  self.data.iloc[self.offset + 0])

    def take_step(self, *args):
        """
        Returns data for current trading day and done signal
        
        """
        obs = self.data.iloc[self.offset + self.step]
        self.step += 1
        done = self.step > self.trading_period
        return obs, done