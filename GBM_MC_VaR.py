import numpy as np
import pandas as pd
import gzip
import math
from pylab import plt
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# unzip, read data into memory-------------------------------------------------
with gzip.open('ohlc.csv.gz') as f:
    df = pd.read_csv(f)

# model functions--------------------------------------------------------------      
def process_data(df):   
    global daily_data 
    
    # separate unique coin data----------------------------------------------------   
    df['date_time'] = pd.to_datetime(df['date_time']) # for resampling, ordering data
    coins = df['asset'].unique()
    
    raw_data = []
    for coin in coins:
        raw_data.append(df[df['asset']==coin])   
        
    # resample to daily bars, derive log returns 
    daily_data = []
    for coin in raw_data:
        coin = coin.set_index('date_time').resample('D', label='right').last() 
        coin['log_rets'] = np.log(coin['close'] / coin['close'].shift()) 
        daily_data.append(coin.dropna())
            
    # quick daily close plot with proper date sequence 
    for coin in daily_data:
        coin['close'].plot(label=coin.iloc[0,1], title='Inspect Price Series', 
                           figsize=[6,4])
        #plt.show()
    plt.legend(loc=0)    
    plt.show()    

def run_simulations(num_sims=1000000):
    global sim_results
   
    # run GBM MC sim on each coin--------------------------------------------------
    sim_results = dict()
    for coin in daily_data: 
        
        # calibrate model  
        def calibrate_MC_model(df):
            # initial price
            S0 = df['close'].iloc[0]
            # annualized returns 
            r = df['log_rets'].mean() * 365
            # annualized volatility
            sigma = df['log_rets'].std() * 365 ** 0.5
            
            return S0, r, sigma
        
        S0, r, sigma = calibrate_MC_model(coin)
        
        # Geometric Brownian Motion Monte Carlo Simulation
        def GBM_MC_simulation(S0, r, sigma):
            '''
            Geomteric Brownian Motion Price Series Simulation
            MODEL PARAMATERS
            ================
            S0 = starting price
            r  = drift (annualized daily return)
            sigma = sigma  (annualized daily volatility)
            '''
            # model constants 
            I = num_sims # simulations
            T = 1/12 # simulation length in years (1 month)
            M = 31 # steps (days) per sim
            dt = T / M
            S = np.zeros((M + 1, I))
            S[0] = S0
            for t in range(1, M + 1):
                S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                        sigma * math.sqrt(dt) * np.random.standard_normal(I))
        
            return S
        
        # capture simulation results
        S = GBM_MC_simulation(S0, r, sigma)
        sim_results[coin.iloc[0,0]] = S
    
    # visualize 100 sims for coin B
    plt.figure(figsize=(6, 4))
    plt.title('100 Sample Sim Paths - Coin B')
    plt.plot(S[:, :100], lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show();

def compare_simulations():
    # compare simulation results---------------------------------------------------
    VaR_results = pd.DataFrame(columns=['VaR_pctl_ret'])
    VaR_pctl = 5 # value-at-risk percentile 
    for key in sim_results.keys():
        
        VaR_pctl_value = np.percentile(sim_results[key][-1], VaR_pctl)
        # plot histogram of ending values
        plt.figure(figsize=(6, 4))
        plt.title(f'1-Month GBM MC Simulation Distribution - Coin {key}')
        plt.hist(sim_results[key][-1], bins=100, label='Simulation Frequency')
        plt.axvline(VaR_pctl_value, c='r', linestyle='--', linewidth=2, 
                     label=f'Value-at-Risk Lower Limit Percentile: {VaR_pctl}')
        plt.xlabel('Ending Price level')
        plt.ylabel('Frequency')
        plt.legend(loc=0)
        plt.show();
        
        # derive % VaR using resultant percentile value and beginning simulation value
        VaR_pctl_ret = VaR_pctl_value / sim_results[key][-0][0] - 1
                
        print(f'Coin {key} 1-month VaR at {VaR_pctl}th percentile: {VaR_pctl_ret*100:.2f}%')            
        VaR_results.loc[key] = VaR_pctl_ret
    
    ranked_Var = VaR_results.sort_values('VaR_pctl_ret', ascending=False)
    print('-'* 76)
    print(f'The best asset under the Monte Carlo value at risk (VaR) criteria is: Coin {ranked_Var.index[0]}')

if __name__ == '__main__':
    
    print('processing_data...')
    process_data(df)
    
    print('running simulations...')    
    run_simulations(num_sims=1000000)
    
    print('comparing simulations...')    
    compare_simulations()





