import numpy as np
import pandas as pd
import re
import warnings
import os
import GP
import GPy

phi_file = os.path.abspath('crimemod1.csv')  # Location of data
buckets = 15  # Number of buckets.

l = [1.82754075018, 1.82754075018, 1.82754075018]
horz = 33.522111

sig_eps_f = lambda train_t: train_t.std()

logTransform = True

file_prefix = 'Results'


def read_data(phi_file):
    ''' Read data '''
    target_type = str  # The desired output type
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")

        phi_data = pd.read_csv(phi_file, sep=",", header=0)
        print("Warnings raised:", ws)
        # We have an error on specific columns, try and load them as string
        for w in ws:
            s = str(w.message)
            print("Warning message:", s)
            match = re.search(r"Columns \(([0-9,]+)\) have mixed types\.", s)
            if match:
                columns = match.group(1).split(',')  # Get columns as a list
                columns = [int(c) for c in columns]
                print("Applying %s dtype to columns:" % target_type, columns)
                phi_data.iloc[:, columns] = phi_data.iloc[
                    :, columns].astype(target_type)

    
    # month, day, year
    date = np.array([x.split('-') for x in phi_data.FROMDATE])
    dom = [int(x) for x in date[:, 0]]
    month = [int(x) for x in date[:, 1]]
    year = [int(x) for x in date[:, 2]]
    # months since Jan 2012
    time_feat = np.subtract(year, 2011) * 12 + month
    
    # grab the features we want
    data_unnorm = np.transpose(
        np.vstack((time_feat, phi_data.X, phi_data.Y))).astype(float)
    # remove NaNs
    good_data = data_unnorm[~(np.isnan(data_unnorm[:, 1]))]
    return good_data

print ("Finished processing data...")

GP.run_gp(read_data(phi_file), buckets, l, horz, sig_eps_f, logTransform,
          file_prefix, 'Philadelphia')