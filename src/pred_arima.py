from statsmodels.tsa.arima_model import ARIMA

'''ARIMA model'''
def model_arima(data):
    
    '''last year's data would be used for training'''
    train_size = int(len(data) -12)
    
    train, test = data[0:train_size], data[train_size:len(data)]
    
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        '''
        p: The number of lag observations included in the model, also called the lag order.
        d: The number of times that the raw observations are differenced, also called the degree of differencing.
        q: The size of the moving average window, also called the order of moving average
        '''
        
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
        
    #print('Test MSE: %.3f' % error)
    return test, predictions 