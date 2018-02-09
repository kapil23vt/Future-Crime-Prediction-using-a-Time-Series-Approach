# Future-Crime-Prediction-using-a-Time-Series-Approach
ECE 5424G Machine Learning Project 

To run this Crime Prediction Software. Please install the latest and compatible versions of the following packages:
Anaconda(contains most of the required packages)
Keras (using Anaconda prompt)
GPy (using pip package manager)
Theano/Tensorflow

pred_RNN.py file contains the main function and when it is executed, it runs the LSTM and ARIMA models.

To run the Gaussian Process, preprocess the csv file(dataset) to form only three coulmns, which are: X(Lat),Y(Long) and From-date(day-month-year). 
Now, run phily_gp.py file, this builds the Gaussian Process Model.

In the above two files, modify the hardcoded file location of the csv file appropriately.
