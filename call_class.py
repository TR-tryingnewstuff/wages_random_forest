#%%
import yfinance as yf
import pandas as pd 
import numpy as np 
import datetime
from hashlib import sha256

PROTOCOL_WALLET = '0xae17142117ree701'
PROTOCOL_TREASURY = 100000

class Wallets():
    def __init__(self) -> None:
       
       self.wallets = {}

    def create_wallet(self):

        id = np.random.choice(int(1e10))

        while id in self.wallets.keys():
            id = np.random.choice(int(1e10))
        
        self.wallets[id] = {
                            'amount': 0,
                            'key': sha256(bytes(id)).hexdigest()
                            }

        return self.wallets[id]

    def transferFrom(self, amount, _from, to):

        if self.amount >= amount:
            self.wallets[_from]['amount'] -= amount
            self.wallets[to]['amount'] += amount

            return True
        
        else:

            print('You are trying to send {} but you only have {}'.format(amount, self.wallets[_from]['amount']))
            return False

    def fund(self, amount, to):

        self.wallets[to]['amount'] += amount

        return self.wallets[to]['amount']

class USCall():

    def __init__(self, underlying: str, amount: int, maturity: np.datetime64, strike: float, price: float, issuer: int, buyer: int) -> None:
        '''Creates an american call option'''

        self.contract = {
                'underlying': underlying,
                'amount': amount, 
                'maturity': maturity,
                'strike': strike,
                'price': price,
                'issuer': issuer,
                'buyer': buyer
        }
        
    def __str__(self):
        '''Returns the call option informations'''

        return str(self.contract)
    
    def redeem(self):
        '''Applies the buyer's right to buy the underlying asset at the strike price resulting in a transfer of strike - current_asset_price'''

        underlying_price = yf.Ticker(self.contract['underlying']).history()['Close'].values[-1]
        outcome = underlying_price - self.contract['strike']

        transferFrom(outcome, )

        return outcome
    
    def transferCall(self, _from: int, to: int):

        if _from == self.buyer:
            self.contract['buyer'] = to

            return True

        else:
            print('you are not the owner of the option')
            return False    
        
    def transferFrom(self, amount, _from, to):

        if _from == self.buyer:
            pass


wallets = Wallets()
new_wallet = wallets.create_wallet()

print(new_wallet)    

print(wallets.wallets)


##call = USCall('AAPL', 10, '2023-10-20', 100, 2, 10224, 1024466)
##print(call)

#print(call.redeem())
# %%
