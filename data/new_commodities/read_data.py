#%% 
import torch 
import pandas as pd 
import numpy as np 

from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r

class Commodities():
    def __init__(self, csv_files, time_range, log_return=False):
        '''
        Each commodity is given by a csv files contaning % Change of 
        prices from 01/01/2010 to 12/31/2020 
        '''

        super(Commodities, self).__init__()

        df0 = 0 
        for idx, file in enumerate(csv_files):
            if log_return is True:
                # read the cdv file 
                df = pd.read_csv(file, delimiter=",", parse_dates=True, thousands=',')

                # add log return column to df 
                df['log_ret'] = np.log(df['Price']) - np.log(df['Price'].shift(1))

                # keep the log_return data only
                df = df.drop(columns=['Change %', 'Price', 'Open', 'High', 'Low', 'Vol.'])
            else:
                df = pd.read_csv(file, delimiter=",", parse_dates=True)

                # extract % change values 
                for i in range(len(df)):
                    df.iloc[i]['Change %'] = df.iloc[i]['Change %'][:-1]

                # keep the % Change column only 
                df = df.drop(columns=['Price', 'Open', 'High', 'Low', 'Vol.'])
                df['Change %'] = df['Change %'].astype(float)

            # take the negative of values 
            # df['Change %'] = -df['Change %'].astype(float)
            

            # select time range
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if time_range is not None:
                df = df[df.index.year.isin(time_range)]

            # get maxima 
            grouped = df.groupby(pd.Grouper(freq='M'))
            grouped_max = grouped.max()

            # merge dataframe 
            if idx == 0:
                df0 = grouped_max
            else:
                df0 = df0.merge(grouped_max, on='Date')     

        # dropna 
        df0 = df0.dropna() 

        # assign data 
        self.data = df0.values

        # assign df 
        self.df = df0 

        # fit margins 
        self.F = np.zeros_like(self.data)

        # GEV params 
        self.gev_params = np.zeros((self.data.shape[1], 3))

        for comdty in range(self.data.shape[1]):
            params = gev.fit(self.data[:, comdty])
            self.gev_params[comdty, :] = params 
            self.F[:, comdty] = gev.cdf(self.data[:, comdty], *params)

        self.F = torch.Tensor(self.F)
        self.data = torch.Tensor(self.data)        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], self.F[idx])

    def emp_cdf(self, threshold):
        prob = (self.data <= threshold).float().prod(1).mean()

        return prob

    def marginal_cdf(self, comdty, threshold):
        params = self.gev_params[comdty, :]
        cdf = gev.cdf(threshold, *params)

        return cdf 

# create commodities dataset 
# csv_files = ['Nickel_Futures_Historical_Data.csv', 'Copper_Futures_Historical_Data.csv', 'Zinc_Futures_Historical_Data.csv'] 
csv_files = ['Coffee.csv', 'Copper.csv', 'Corn.csv', 'Crude_oil.csv', 'Gold.csv', 'Heating_oil.csv', 'Natural_Gas.csv', 'Platinum.csv', 'silver.csv', 'Wheat.csv']
# dataset = Commodities(csv_files=csv_files)        
dataset = Commodities(csv_files=[csv_files[2]], time_range=[2015, 2017], log_return=False)

# read data 
df = pd.read_csv(csv_files[2], delimiter=",", parse_dates=True, thousands=',')

# add log return column to df 
df['log_ret'] = np.log(df['Price']) - np.log(df['Price'].shift(1))

# keep the log_return data only
df = df.drop(columns=['Change %', 'Price', 'Open', 'High', 'Low', 'Vol.'])

# dropna 
df = df.dropna() 

# Augmented dickey-Fuller test
from statsmodels.tsa.stattools import adfuller 
result = adfuller(df['log_ret'].values)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# # read from the commodities data 
# file = 'Nickel_Futures_Historical_Data.csv'
# df = pd.read_csv(file, delimiter=",", parse_dates=True)
# # extract % change values 
# for i in range(len(df)):
#     df.iloc[i]['Change %'] = df.iloc[i]['Change %'][:-1]

# # keep the % Change column only 
# df = df.drop(columns=['Price', 'Open', 'High', 'Low', 'Vol.'])

# # transform columns to float
# # We are interested in P[min return < threshold], therefore we 
# # take the negative values of return and do EVT on the daily/monthly maximum 
# # of these values 
# df['Change %'] = -df['Change %'].astype(float)
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)
# grouped = df.groupby(pd.Grouper(freq='M'))
# grouped_max = grouped.max()
# grouped_max = grouped_max.values 

# # fit GEV to the values 
# params = gev.fit(grouped_max)

# # check survival prob for Nickel 
# threshold = 11.0
# model_survival = 1-gev.cdf(threshold, params[0], params[1], params[2])
# emp_survival = np.mean(grouped_max > threshold) 
# %%
