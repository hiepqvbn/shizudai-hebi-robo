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
            print(self.log_fr)
            print((self.log_fr.columns == ['theta','camera']).all())
            if not (self.log_fr.columns == ['theta','camera']).all():   ##bug here
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
        col=pd.Series(['timestamp', 'theta', 'camera'])
        dt_fr = pd.DataFrame(columns=col)
        dt_fr = dt_fr.set_index('timestamp')
        return dt_fr

    def get_now(self):
        return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def write_data(self, dt):
        print(self.log_fr)
        self.log_fr.loc[self.get_now()] = dt

    def save_dataframe(self):
        print(self.log_fr)
        self.log_fr.to_csv(self.csv_filename)

def main():
    dt = Data_Collect()
    dt.write_data([33,59])
    dt.save_dataframe()


if __name__=='__main__':
    main()

# print(np.load('save_data.npy'))
