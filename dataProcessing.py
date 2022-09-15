import numpy as np
import pandas as pd
import d3rlpy

class Dateset():
    def __init__(self,filename):
        self.filename=filename

    def get_observations(self):
        raw_data=pd.read_csv(self.filename)
        col_n=['Close']
        dataset=pd.DataFrame(raw_data,columns=col_n)
        return dataset.to_numpy().reshape(1,-1)

    def get_actions(self):
        actions=np.random.uniform(low=-1, high=1, size=1)
        return actions

    def get_rewards(self):
        return np.diff(self.get_observations(),n=1)

    def MDPDateset(self):
        terminals = np.random.randint(2, size=len(self.get_observations()))
        dataset = d3rlpy.datasets.MDPDataset(
            observations=self.get_observations(),
            actions=self.get_actions(),
            rewards=self.get_rewards(),
            terminals=terminals
        )
        return dataset


