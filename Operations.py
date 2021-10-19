###This class is defined to use the methods from linearregression.py directly.
import pandas as pd
import numpy as np
import pickle
from linearRegression import LinearRegressionModel as lr
from NanDealer import nanDealer as nd

lr=lr()
nd=nd()
class model_Operations():
    def createDataFrame(self):
        try:
            filename=str(input("Enter the file name\n"))
            dataframe=lr.createDataframe(filename=filename)
            print("The dataframe = \n",dataframe)
            return dataframe
        except Exception as e:
            raise Exception("Error occured while creating the dataframe\n",str(e))

    def fetchingColumns(self,dataframe):
        try:
            columns=lr.showColumns(dataframe=dataframe)
            for i in columns:
                print(i)
            return columns
        except Exception as e:
            raise Exception("Error occured while fetching the columns of the dataframe\n",str(e))

    def fillNanValuesInDataframe(self,dataframe):
        try:
            choice=int(input("Enter the choice for filling og the nan values in the dataframe \n1. Fill Nan values with column Mean"
                             "2. Fill Nan values with column median"
                             "3. Fill Nan values with rows Mean"
                             "4. Fill Nan values with rows Median"
                             "5. Fill Nan values with random value from column\n\n"))
            if choice == 1:
                updated_dataframe=nd.fillNanWithMeanColumnWise(dataframe)
            elif choice == 2:
                updated_dataframe = nd.fillNanWithMedianColumnWise(dataframe)
            elif choice == 3:
                updated_dataframe = nd.fillNanWithMeanRowWise(dataframe)
            elif choice == 4:
                updated_dataframe = nd.fillNanWithMedianRowWise(dataframe)
            elif choice == 5:
                updated_dataframe = nd.fillNanWithRandomValuesFromColumn(dataframe)
            else:
                print("Invalid choice entered for Nan values completion, kindly choose between 1 and 5")
            print("Data frame after filling the nan values =\n",updated_dataframe)
            return updated_dataframe
        except Exception as e:
            raise Exception("Error occured while filling Nan values\n",str(e))
    def splitData(self,dataframe):
        try:
            ratio=float(input("Enter the ratio of split- for eg 0.2 for 20 % train data and 80% test data\n"))
            test_data,train_data=lr.split_train_test(dataframe,ratio)
            print("Test data =\n",test_data)
            print("Train data=\n",train_data)
            return train_data,test_data
        except Exception as e:
            raise Exception("Error occured while fetching the columns of the dataframe\n", str(e))

    def pandasProfiling(self,train_data):
        try:
            htmlFileName = str(input("Enter the file name\n"))
            profile = lr.pandasProfiling(htmlFileName, train_data)
            print(profile)
            return profile
        except Exception as e:
            raise Exception("Error occured while pandas profiling\n",str(e))

    def modelPreparations(self):
        try:
            features, label = [], []
            nx = int(input("Enter size of features required in x axis\n"))
            ny = int(input("Enter size of label field required in y axis\n"))
            for i in range(0, nx):
                ele = input("Enter the {0} feature".format(i + 1))
                features.append(ele)
            for i in range(ny):
                ele = input("Enter the {0} label".format(i + 1))
                label.append(ele)
            print("features list = ",features)
            print("label = ",label)

            return features,label
        except Exception as e:
            raise Exception("Error occured while taking input for model preparations\n",str(e))

    def scalingDataframe(self,test_data,features):
        try:
            scaled_features_DF, scaled_features = lr.featureScaling(test_data[features])
            scaled_features_DF.columns = features
            print("Scaled features in dataframe\n",scaled_features_DF)
            return scaled_features_DF, scaled_features
        except Exception as e:
            raise Exception("Error occured while scaling the features\n",str(e))

    def buildModel(self,train_data,features,label):
        try:
            ##We can use if condition here to define the scalar values model to be built

            # linear, coef, intercept = lr.buildModel(train_data[features], train_data[label])

            linear, coef, intercept = lr.buildModel(train_data[features], label)
            print("attained slope=", coef)
            print("attained intercept=", intercept)
            return linear,coef,intercept
        except Exception as e:
            raise Exception("Error occured while building the model\n",str(e))

    def pickleTheModel(self,modelInstance):
        try:
            filename = str(input("Enter the file name to dump the model using pickle in .sav format\n"))
            lr.saveModel(modelInstance, filename)
            print("Model saved in pickle format")
            return filename
        except Exception as e:
            raise Exception("Error occured while pickling the file\n",str(e))

    def predictingLabelValue(self,features,modelInstance):
        try:
            feature_values = []
            for i in range(0, len(features)):
                ele = float(input("Enter the {0} feature value".format(i + 1)))
                feature_values.append(ele)
            print(feature_values)
            decision=str(input("Do you want to scale your input values?? enter yes or no\n"))
            if decision=='Yes' or decision == 'yes':
                # converting the entered values into scalar transformation
                # transposing the values using dataframe
                feature_values=pd.DataFrame(feature_values)
                # print(type(feature_values))
                testing_feature_df,testing_feature_arr=lr.featureScaling(feature_values)
                # print(testing_feature_arr)
                feature_values=testing_feature_arr.flatten().tolist()
                # print(feature_values)
            predicted_value= lr.predicting(modelInstance, feature_values)

            print("Prediction = ", predicted_value)
            return predicted_value,feature_values
        except Exception as e:
            raise Exception("Error occured while taking input values for prediction\n",str(e))

    def checkAccuracy(self,modelInstance,train_data,feature,label):
        try:
            ## To calculate the r square value
            accuracy = lr.CheckAccuracy(modelInstance, train_data[feature], label)
            print("Accuracy of the model is =", accuracy * 100)
            return accuracy
        except Exception as e:
            raise Exception("Error occured while checking the accuracy\n",str(e))

    def checkVIF(self,scaled_features,features,dataframe):
        try:
            vif_df = lr.checkVIF(scaled_features, dataframe[features])
            print("Scaled features VIF = \n", vif_df)
            return vif_df
        except Exception as e:
            raise Exception("Error occured while checking the VIF\n",str(e))

    def calculatedAdjustedR2(self,train_data,features,rsqaure):
        try:
            adjused_r2 = lr.calculateAdjustedR2(train_data[features],rsqaure)
            #adjused_r2=adjused_r2*100
            print("The adjusted r square value = ", adjused_r2)
            return adjused_r2
        except Exception as e:
            raise Exception("Error occured while calculating the adjusted r square\n",str(e))

    def lasso(self,test_data,train_data,features,label):
        try:
            cv=int(input("Enter the cross validation value which you want for Lasso"))
            max_iter=int(input("Enter the maximum iterations you want for the lasso operation"))
            alpha,score=lr.performLassoOperations(cv,max_iter,train_data[features],train_data[label],test_data[features],test_data[label])
            print("The alpha used in Lasso was=\n",alpha)
            print("The score using lasso is =\n",score)

            return alpha,score

        except Exception as e:
            raise Exception("Error occured while taking input for lasso\n", str(e))

    def ridge(self,test_data,train_data,features,label):
        try:
            cv = int(input("Enter the cross validation value which you want for Ridge"))

            alpha, score = lr.perfromRidgeOperations(cv,train_data[features], train_data[label],
                                                     test_data[features], test_data[label])
            print("The alpha used in Ridge was=\n", alpha)
            print("The score used in Ridge is =\n", score)

            return alpha, score

        except Exception as e:
          raise Exception("Error occured while taking input for ridge\n", str(e))


    def elasticNet(self,train_data,test_data,features,label):
        try:
            alpha_values=None
            cv = int(input("Enter the cross validation value which you want for ElasticNet"))
            alpha, l1_ratio, score=lr.performElasticNetOperations(alpha_values,cv,train_data[features],train_data[label],test_data[features],test_data[label])
            print(f"The alphas ,l1_ratio and score values for elasticNet is {alpha,l1_ratio,score}")

            return alpha,l1_ratio,score

        except Exception as e:
             raise Exception("Error occured while taking input for elastic\n", str(e))

    def predictingFromSavedModel(self,filename,feature_values):
        try:
            loaded_model = pickle.load(open('ai4i2020.sav', 'rb'))
            predicted = loaded_model.predict([feature_values])
            print(predicted)
            return predicted
        except Exception as e:
            raise Exception("Error occured while prediction from the saved model\n",str(e))