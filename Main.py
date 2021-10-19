from Operations import model_Operations
import pandas as pd

##Creating the class instance for model_operations

mo=model_Operations()

dataframe=mo.createDataFrame()
columns=mo.fetchingColumns(dataframe)
updated_dataframe=mo.fillNanValuesInDataframe(dataframe)
train_data,test_data=mo.splitData(updated_dataframe)
#profile=mo.pandasProfiling(train_data)
features,label=mo.modelPreparations()
#scaled_features_df,scaled_features=mo.scalingDataframe(train_data,features)
linear,coeff,intercept=mo.buildModel(train_data,features,train_data[label])
savedModel=mo.pickleTheModel(linear)
predicted_value,feature_input=mo.predictingLabelValue(features,linear)
# predicted_savedModel=mo.predictingFromSavedModel(savedModel,feature_input)
rSquare=mo.checkAccuracy(linear,train_data,features,train_data[label])
#vif_df=mo.checkVIF(scaled_features,features,dataframe)
adjustedRSquare=mo.calculatedAdjustedR2(train_data,features,rSquare)
alpha_lasso,score_lasso=mo.lasso(test_data,train_data,features,label)
alpha_ridge,score_ridge=mo.ridge(test_data,train_data,features,label)
alpha_elastic, l1_ratio, score_elastic=mo.elasticNet(train_data,test_data,features,label)

'''
btn btn-primary btn-lg
'''