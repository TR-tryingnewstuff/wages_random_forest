#%%
import yfinance as yf
import pandas as pd 
import numpy as np 
from hashlib import sha256


class Wallets():
    def __init__(self) -> None:
       
       self.wallets = {}

    def create_wallet(self):

        id = np.random.choice(int(1e10))
        key = sha256(bytes(id)).hexdigest()

        while id in self.wallets.keys():
            id = np.random.choice(int(1e10))
            key = sha256(bytes(id)).hexdigest()
        
        
        self.wallets[key] = {
                            'amount': 0, 
                            }

        return key

    def transferFrom(self, amount, _from, to):

        if self.wallets[_from]['amount'] >= amount:
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
        '''Creates an american call option contract'''

        wallets.transferFrom(price, buyer, issuer)
        

        self.contract = {
                'underlying': underlying,
                'amount': amount, 
                'maturity': maturity,
                'strike': strike,
                'issuer': issuer,
                'buyer': buyer
        }

        wallets.wallets[buyer]['USCall_{}'.format(self.contract['underlying'])] = 1

    def __str__(self):
        '''Returns the call option informations'''

        return str(self.contract)
    
    def redeem(self):
        '''Applies the buyer's right to buy the underlying asset at the strike price resulting in a transfer of strike - current_asset_price'''

        dict_key = 'USCall_{}'.format(self.contract['underlying'])

        if wallets.wallets[self.contract['buyer']][dict_key] > 0:
            underlying_price = yf.Ticker(self.contract['underlying']).history()['Close'].values[-1]
            outcome = underlying_price - self.contract['strike']

            wallets.wallets[self.contract['buyer']][dict_key] -= 1
            wallets.transferFrom(outcome, self.contract['issuer'], self.contract['buyer'])

            return outcome
        
        else:
            print('you do not own any option')

            return False
        
    def transferCall(self, to: int):
        '''Transfer the call option'''

        dict_key = 'USCall_{}'.format(self.contract['underlying'])

        if wallets.wallets[self.contract['buyer']][dict_key] > 0:

            wallets.wallets[self.contract['buyer']][dict_key] -= 1
            self.contract['buyer'] = to
            wallets.wallets[to][dict_key] = 1

            return True

        else:
            print('you are not the owner of the option')
            return False    
        


wallets = Wallets()

# --------------------------------- Launching the defi protocol -----------------------------------------------

PROTOCOL_WALLET = wallets.create_wallet()
PROTOCOL_TREASURY = 100000
wallets.fund(PROTOCOL_TREASURY, PROTOCOL_WALLET)

# ---------------------------------- creating and funding clients wallets -------------------------------------

thomas_wallet = wallets.create_wallet()
wallets.fund(100, thomas_wallet)

maxime_wallet = wallets.create_wallet()

# ---------------------------------- buying / transferring / redeeming call option ----------------------------

call = USCall('AAPL', 10, '2023-10-20', 100, 2, PROTOCOL_WALLET, thomas_wallet)

print(f'characteristics of the call option contract :  {call}')
call.transferCall(maxime_wallet)

print(f'the buyer of the option has changed after the transfer :  {call}')

print(f'Maxime now owns the call option after it was transferred to him : {wallets.wallets[maxime_wallet]}')

print(f'Maxime decide to exercise the option for a profit of : {call.redeem()}')
print(f'Maxime doesn\'t own the option anymore but he has received the profit from the option : {wallets.wallets[maxime_wallet]}')

# %%
