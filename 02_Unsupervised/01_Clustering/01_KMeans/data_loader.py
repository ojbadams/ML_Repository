import pandas as pd
import numpy as np

class UnsupervisedDataLoader():
    def __init__(self, filename: str, header: int = None, indicator_column: str = None, columns_to_use: list = None):
        self.data = pd.read_csv(filename, header = header)
        self.x = self.data

        if indicator_column is not None:
            self.x = self.data.drop(columns = [indicator_column])   

        if columns_to_use is not None:
            self.x = self.x[columns_to_use]

    def get_data(self): 
        return np.array(self.x)


