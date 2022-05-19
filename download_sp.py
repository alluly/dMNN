'''
Helper function to download the S&P data from AlphaVantage. 

Requires API key given by api variable.
'''
import requests
import time 

api = 'YOUR KEY'

symbols_list = ''
with open('data/sp.txt','r') as f:
    symbols_list = f.read()

symbols_list = [s for s in symbols_list.split('\n') if len(s) > 0]

for symbol in symbols_list[1::2][:-15]:
    print(symbol)

    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full&apikey={}&datatype=csv'.format(symbol, api)

    response = requests.get(url)
    csv = response.text
    with open('data/sp_data/{}.csv'.format(symbol),'w') as f:
        f.write(csv)
    time.sleep(20)
    print('========')
