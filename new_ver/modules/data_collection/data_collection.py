import numpy as np
import pandas as pd
from datetime import date, datetime


class DataCollector:
    def __init__(self, cols=[]):
        self.col = pd.Index(['timestamp'] + cols)
        self.log_df = self.load_dataframe()

    def load_dataframe(self):
        try:
            log_df = pd.read_csv(self.csv_filename)
            if not log_df.columns.equals(self.col):
                log_df = self.make_new_dataframe()
        except FileNotFoundError:
            log_df = self.make_new_dataframe()
        return log_df.set_index('timestamp')

    def make_new_dataframe(self):
        """
        TABLE:
        timestamp|theta0|theta1|theta2|...|camera_x|camera_y|
                 |      |      |      |   |        |        |
        """
        return pd.DataFrame(columns=self.col)

    def get_now(self):
        return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

    def write_data_to_dataframe(self, dt):
        self.log_df.loc[self.get_now()] = dt

    def save_dataframe(self):
        self.log_df.to_csv(self.csv_filename)

    def write_data_to_csv(self, dt):
        self.write_data_to_dataframe(dt)
        self.save_dataframe()

    def clear_csv(self):
        self.make_new_dataframe().to_csv(self.csv_filename, index=False)


class SimulationDataCollector(DataCollector):
    def __init__(self, cols=[]):
        self.csv_filename = f'data/simulations/datalog/datalog{date.today()}.csv'
        super().__init__(cols)


class ExperimentDataCollector(DataCollector):
    def __init__(self, cols=[]):
        self.csv_filename = f'data/experiments/datalog{date.today()}.csv'
        super().__init__(cols)
