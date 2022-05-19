'''
Helper function to download the cryptocurrency data.
'''

import requests
import time 
import pandas as pd
from cryptocmd import CmcScraper

df = pd.read_csv('data/digital_currency_list.csv')
symbols_list = df['currency_code'].values

for symbol in symbols_list[5::2]:
    print(symbol)

    try:
        scraper = CmcScraper(symbol)

        # export the data as csv file, you can also pass optional `name` parameter
        scraper.export("csv", name="data/crypto/{}_all_time".format(symbol))
        time.sleep(3)
        print('========')
    except:
        continue
