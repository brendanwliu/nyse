from data_technicals_methods import *
from collections import deque
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

dataset_stock = prep_data('GS')
#using the fft transform to get out the key frequencies of our stock prices
def fft_transform(dataset):
    close_fft = np.fft.fft(np.asarray(dataset['close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    return fft_df

def plotfft(fft_ds):
    '''plot our fft transforms, with different compenents added to our function'''
    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_ds['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    plt.plot(dataset_stock['close'],  label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()

    items = deque(np.asarray(fft_ds['absolute'].tolist()))
    items.rotate(int(np.floor(len(fft_ds)/2)))
    plt.figure(figsize=(10,7), dpi= 80)
    plt.stem(items)
    plt.title("Figure 4: Components of Fourier Transforms")
    plt.show()

fft_transform(dataset_stock)
#Plotting our fft_transforms but let's only do this once to see how our fft_transforms look like, we can regraph on main.py
#plotfft(fft_transform(dataset_stock))

def ARIMA_analysis(dataset_stock, test_size = 0.66):
    '''Using ARIMA as a feature of our stock price movements. p: The number of lag observations included in the model, also called the lag order.
    d: The number of times that the raw observations are differenced, also called the degree of differencing.
    q: The size of the moving average window, also called the order of moving average.'''
    series = dataset_stock['close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    #this is our autocorrelation plot
    autocorrelation_plot(series)
    plt.figure(figsize=(10, 7), dpi=80)
    plt.show()
    #Get our ARIMA analysis
    X = series.values
    size = int(len(X) * test_size)
    train, test = X[0:size], X[size:len(X)] #prep data into training and testing
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return series, test, predictions

# Plot the predicted (from ARIMA) and real prices this process is a little slow, no need
# to run this more than once to observe how well our ARIMA model works against our data
def ARIMA_plot(test, predictions):
    plt.plot(test, label='Real')
    plt.plot(predictions, color='red', label='Predicted')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 5: ARIMA model on GS stock')
    plt.legend()
    plt.show()
