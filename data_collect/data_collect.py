import numpy as np
import pandas as pd

from datetime import date, datetime

class Data_Collect(object):

    def __init__(self) -> None:
        
        #csv file name = datalog+today.csv
        self.csv_filename = 'datalog/datalog{}.csv'.format(date.today())

        #load csv file as pandas dataframe
        try:    
            self.log_fr = pd.read_csv(self.csv_filename)
            if not (self.log_fr.columns == ['theta','camera']).all():
                print("ok")
                self.log_fr = self.make_new_dataframe()
        except:     #if data file haven't made, make as an empty csv
            self.log_fr = self.make_new_dataframe()

    
    def make_new_dataframe(self):
        """
        TABLE:
        timestamp|theta0|theta1|theta2|...|camera_x|camera_y|
                 |      |      |      |   |        |        |
        """
        # timestamp_series = pd.Series()
        dt_fr = pd.DataFrame(columns=['theta','camera','timestamp'])
        dt_fr = dt_fr.set_index('timestamp', )
        return dt_fr

    def get_now(self):
        return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def write_data(self, dt):
        self.log_fr.loc[self.get_now()] = dt

    def save_dataframe(self):
        self.log_fr.to_csv(self.csv_filename)

def main():
    dt = Data_Collect()
    print(dt.log_fr)
    dt.save_dataframe()


if __name__=='__main__':
    main()

# print(np.load('save_data.npy'))
