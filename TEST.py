import pickle
import pandas as pd
from Operations import model_Operations as mo
from linearRegression import LinearRegressionModel as lr
import pickle
import numpy as np

o=lr()
# filename=str(input("Enter the file name with .sav"))
filename='Linearregression_example.pkl'
loaded_model=pickle.load(open(filename,'rb'))
feature_values=np.array([309.3,1551,42.8])
#feature_values=pd.DataFrame(feature_values)
# testing_feature_df,testing_feature_arr=o.featureScaling(feature_values)
#feature_values = testing_feature_arr.flatten().tolist()
print(feature_values)
predicted=loaded_model.predict([feature_values])
print(predicted)
