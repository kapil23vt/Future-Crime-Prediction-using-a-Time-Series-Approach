import numpy
import pandas
import re
import warnings
import sys
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import util
import pred_arima


''' global variables '''
n_buckets = 15  # Number of buckets.


'''LSTM model array for each square in the grid'''
model_lstm = []
for i in range(n_buckets):
    model_lstm.append([])
    for j in range(n_buckets):
        '''LSTM layers would be sequentially arranged'''
        model_lstm[i].append(Sequential())
        

def read_data(phily_file):
    ''' Read data '''
    target_type = str  # The desired output type
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")

        phily_data = pandas.read_csv(phily_file, sep=",", header=0,  low_memory=False)
        #print("Warnings raised:", ws)
        # We have an error on specific columns, try and load them as string
        for w in ws:
            s = str(w.message)
            print("Warning message:", s)
            match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
            if match:
                columns = match.group(1).split(',')  # Get columns as a list
                columns = [int(c) for c in columns]
                print("Applying %s dtype to columns:" % target_type, columns)
                phily_data.iloc[:, columns] = phily_data.iloc[
                    :, columns].astype(target_type)

    # Month is in the form of year-month
    date = numpy.array([x.split('-') for x in phily_data.Month])
    
    month = [int(x) for x in date[:, 1]]
    year = [int(x) for x in date[:, 0]]
      
    #number of months since Jan 2011
    time_feat = numpy.subtract(year, 2011) * 12 + month
  
    # grab Lat,Long and time_feat
    data_unnorm = numpy.transpose(
        numpy.vstack((time_feat, phily_data.Lat, phily_data.Lon))).astype(float)
        
    # remove NaNs
    good_data = data_unnorm[~(numpy.isnan(data_unnorm[:, 1]))]
    #print(np.min(good_data[:,0]))
    print ("Finished processing Philadelphia crime data...")
    
    return good_data


# convert an array of values into a dataset matrix in the form of ...t-3,t-2,t-1,t
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
 
 
#LSTM MODEL-which gets trained and then predicts crime count
def pred_lstm(data,row,column):
    
    # assign a random seed
    numpy.random.seed(7)
    
    # LSTMs are sensitive to the scale of data, it will be efficient to normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    # split the data into training and test data, we use last year's data for testing
    train_size = int(len(data) -12)
    
    train, test = data[0:train_size,:], data[train_size:len(data),:]    
    
    # reshape the data into X=month and Y=month+1, so that we form a time series
    look_back = 8# no of previous timesteps we use to predict the data
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # The LSTM network expects the input data (X) to be provided with a specific array structure in the form of: [samples, time steps, features].
    #reshape the data to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    
    # create and fit the LSTM network
    #The batch_size is the number of samples from your train dataset shown to the model at a time.
    batch_size = 1
    
    #LSTM networks can be stacked in Keras in the same way that other layer types can be stacked. 
    #One addition to the configuration that is required is that an LSTM layer prior to each subsequent LSTM layer must return the sequence. 
    #This can be done by setting the return_sequences parameter on the layer to True.
    model_lstm[row][column].add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model_lstm[row][column].add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model_lstm[row][column].add(Dense(1))
    model_lstm[row][column].compile(loss='mean_squared_error', optimizer='adam')
    
    for i in range(100):
        model_lstm[row][column].fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        #reset after each exposure to data
        model_lstm[row][column].reset_states()
    
    #Do predictions using the model
    trainPredict = model_lstm[row][column].predict(trainX, batch_size=batch_size)
    model_lstm[row][column].reset_states()
    testPredict = model_lstm[row][column].predict(testX, batch_size=batch_size)
    
    #invert the predictions back to normal scale
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    
    return testY[0], testPredict[:,0]
    
    '''
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Training error: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Testing Error: %.2f RMSE' % (testScore))
    
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(data)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(data)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
    
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(data))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    '''
    

        
def main():
    test_lstm = []
    test_arima = []
    predictions_lstm = []
    predictions_arima = []
    
    good_data = read_data('D:\Spring-2017\DBMS\Project\philadelphiacrimedata\crime_3.csv')
    
    '''create the data into buckets'''
    grid_buckets = util.createBuckets(good_data, n_buckets)
   
    min_time_feat = sys.maxsize
    max_time_feat = -sys.maxsize
    for i in range(len(grid_buckets)):
        if grid_buckets[i][0] < min_time_feat:
            min_time_feat = grid_buckets[i][0]
        if grid_buckets[i][0] > max_time_feat:
            max_time_feat = grid_buckets[i][0]
    
    '''find min and max time_feat value'''
    max_time_feat = int(max_time_feat)
    min_time_feat = int(min_time_feat)
   
    data_grid = numpy.zeros((n_buckets,n_buckets,max_time_feat-min_time_feat+1))
    
    for row in range(len(grid_buckets)):
        data_grid[int(grid_buckets[row][1])][int(grid_buckets[row][2])][int(grid_buckets[row][0])-min_time_feat] = float(grid_buckets[row][3]) 
    
    '''for each square in the grid do below'''
    for i in range(n_buckets):
        for j in range(n_buckets):
            data_per_grid = data_grid[i][j]
            
            if numpy.count_nonzero(data_per_grid) == 0:
                continue
            
            tempTest, tempPredictions = pred_arima.model_arima(data_per_grid)       
            test_arima = numpy.append(test_arima,tempTest)
            predictions_arima = numpy.append(predictions_arima,tempPredictions)
            
            data_per_grid = numpy.reshape(data_per_grid,(len(data_per_grid),1))
            tempTest, tempPredictions = pred_lstm(data_per_grid,i,j)
            test_lstm = numpy.append(test_lstm,tempTest)
            predictions_lstm = numpy.append(predictions_lstm,tempPredictions)
            
        

    predictions_arima[predictions_arima < 0] = 0        
    error = math.sqrt(mean_squared_error(test_arima, predictions_arima))
    mea = mean_absolute_error(test_arima,predictions_arima)
    r2 = r2_score(test_arima,predictions_arima)
    print('ARIMA:RMSE: %.3f' % error)
    print('ARIMA: Mean Absolute Error:')
    print(mea)
    print('R^2_score is:')
    print(r2)
    
    plt.plot(test_arima,color = 'blue',label='Actual')
    plt.plot(predictions_arima, color='red',label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel("Month")
    plt.ylabel("Count per Month")
    plt.xticks([])
    plt.title('Crime Prediction using ARIMA')
    plt.show()
    
    
    predictions_lstm[predictions_lstm < 0] = 0
    error = math.sqrt(mean_squared_error(test_lstm, predictions_lstm))
    mea = mean_absolute_error(test_lstm,predictions_lstm)
    r2 = r2_score(test_lstm,predictions_lstm)
    print('LSTM: RMSE: %.3f' % error)
    print('Mean Absolute Error:')
    print(mea)
    print('R^2_score is:')
    print(r2)

    plt.plot(test_lstm,color = 'blue',label='Actual')
    plt.plot(predictions_lstm, color='red',label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel("Month")
    plt.ylabel("Count per Month")
    plt.xticks([])
    plt.title('Crime Prediction using LSTM')
    plt.show()
    
    
if __name__ == '__main__':
    main()    