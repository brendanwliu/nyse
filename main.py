from data_technicals_methods import *
from FT_stock_analysis import *
import argparse
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi


dataset_stock = prep_data('GS')
dataset_TI = get_technicals_indic(dataset_stock)
plotting_TI(dataset_TI,400)
