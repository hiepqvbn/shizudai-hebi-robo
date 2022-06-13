import numpy as np
import pandas as pd

from datetime import date, datetime

class DataCollect(object):

    col=pd.Series(['timestamp', 'theta', 'camera'])

    def __init__(self) -> None:
        
        #csv file name = datalog+today.csv
        self.csv_filename = 'datalog/datalog{}.csv'.format(date.today())

        #load csv file as pandas dataframe
        try:    
            self.log_df = pd.read_csv(self.csv_filename)
            # print(self.log_df)
            # print((self.log_df.columns == ['theta','camera']).all())
            if not (self.log_df.columns == self.col).all():   ##bug here
                # print("ok")
                self.log_df = self.make_new_dataframe()
        except:     #if data file haven't made, make as an empty csv
            self.log_df = self.make_new_dataframe()
        finally:
            self.log_df = self.log_df.set_index('timestamp')

    
    def make_new_dataframe(self):
        """
        TABLE:
        timestamp|theta0|theta1|theta2|...|camera_x|camera_y|
                 |      |      |      |   |        |        |
        """
        # timestamp_series = pd.Series()
        
        dt_fr = pd.DataFrame(columns=self.col)
        
        # print((dt_fr.columns == ['theta','camera']).all())
        return dt_fr

    def get_now(self):
        return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def write_data(self, dt):
        # print(self.log_df)
        
        self.log_df.loc[self.get_now()] = dt

    def save_dataframe(self):
        # print(self.log_df)
        self.log_df.to_csv(self.csv_filename)

    def clear_csv(self):
        self.make_new_dataframe().set_index('timestamp').to_csv(self.csv_filename)

def main():
    dt = DataCollect()
    dt.write_data([33,59])
    dt.save_dataframe()


if __name__=='__main__':
    main()

# print(np.load('save_data.npy'))
