import pandas as pd
import pickle
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet, Ridge, Lasso, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class LinearRegressionModel:
    def createDataframe(self, filename):
        try:
            df = pd.read_csv(filename)
            return df
        except Exception as e:
            raise Exception("Error orrcured while loading the dataframe", str(e))

    def split_train_test(self, data, test_ratio):
        try:
            np.random.seed(10)
            shuffled_indices = np.random.permutation(len(data))
            total_set_size = int(len(data) * test_ratio)
            test_indices = shuffled_indices[:total_set_size]
            train_indices = shuffled_indices[total_set_size:]
            return data.iloc[train_indices], data.iloc[test_indices]
        except Exception as e:
            raise Exception("Error occured while splitting the data frame for test and train", str(e))

    def pandasProfiling(self, htmlFileName, dataframe):
        try:
            profile = ProfileReport(dataframe)
            profile.to_file(htmlFileName)
            return profile
        except Exception as e:
            raise Exception("Error orrcured while profiling the data the dataframe", str(e))

    def showColumns(self, dataframe):
        try:
            return dataframe.columns
        except Exception as e:
            raise Exception("Error occured while fetching the column names from csv", str(e))

    def featureScaling(self, featuresList):
        try:
            scaler = StandardScaler()
            feature_array = scaler.fit_transform(featuresList)
            return pd.DataFrame(feature_array), feature_array
        except Exception as e:
            raise Exception("Error occured while scaling the features", str(e))

    def checkVIF(self, array, user_choice_features_df):
        try:
            vif_df = pd.DataFrame()
            vif_df['VIF'] = [variance_inflation_factor(array, i) for i in range(array.shape[1])]
            vif_df['Features'] = user_choice_features_df.columns
            return vif_df
        except Exception as e:
            raise Exception("Error occured while checking the VIF", str(e))

    def buildModel(self, listOfFeatures, labelValue):
        try:
            #             x=dataframe[listOfFeatures]
            #             y=dataframe[labelValue]
            x = listOfFeatures
            y = labelValue
            linear = LinearRegression()
            linear.fit(x,y)
            coef = linear.coef_
            intercept = linear.intercept_
            return linear, coef, intercept
        except Exception as e:
            raise Exception("Error orrcured while buildling the model", str(e))

    def saveModel(self, linear, fileName):
        try:
            pickle.dump(linear, open(fileName, 'wb'))
        except Exception as e:
            raise Exception("Error orrcured while saving the model", str(e))

    def loadingTheModel(self, fileName):
        try:
            saved_model = pickle.load(open(fileName, 'rb'))
            return saved_model
        except Exception as e:
            raise Exception("Error orrcured while loading the model", str(e))

    def predicting(self, linear, featureValues):
        try:
            return linear.predict([featureValues])
        except Exception as e:
            raise Exception("Error occured while predicting the label value", str(e))

    def CheckAccuracy(self, linear, listOfFeatures, labelValue):
        try:
            #             x=dataframe[listOfFeatures]
            #             y=dataframe[labelValue]
            x = listOfFeatures
            y = labelValue
            score = linear.score(x, y)

            return score
        except Exception as e:
            raise Exception("Error occured while calculating the accuracy of the model", str(e))

    def calculateAdjustedR2(self, features_df, r2_value):
        try:
            p = features_df.shape[1]
            n = features_df.shape[0]
            adjusted_r2 = 1 - (1 - r2_value) * (n - 1) / (n - p - 1)
            adjusted_r2 = adjusted_r2 * 100
            return adjusted_r2
        except Exception as e:
            raise Exception("Error occured while calculating the adjusted r sqaure", str(e))

    def performLassoOperations(self, cv, max_iter, train_data_features, train_data_label, test_data_features,
                               test_data_label):
        try:
            ## performing lasso cross validation
            lassocv = LassoCV(cv=cv, max_iter=max_iter, normalize=True)
            lassocv.fit(train_data_features, train_data_label)
            alpha = lassocv.alpha_
            ##Building regression model with lasso
            lasso = Lasso(alpha)
            lasso.fit(train_data_features, train_data_label)
            score = lasso.score(test_data_features, test_data_label)

            return alpha, score
        except Exception as e:
            raise Exception("Error occured while performing lasso operation", str(e))

    def perfromRidgeOperations(self, cv, train_data_features, train_data_label, test_data_features, test_data_label):
        try:
            ## performing ridge cross validation
            ridgecv = RidgeCV(cv=cv, normalize=True)
            ridgecv.fit(train_data_features, train_data_label)
            alpha = ridgecv.alpha_
            ##Building the ridge model
            ridge = Ridge(alpha)
            ridge.fit(train_data_features, train_data_label)
            score = ridge.score(test_data_features, test_data_label)

            return alpha, score

        except Exception as e:
            raise Exception("Error occured while performing ridge operation", str(e))

    def performElasticNetOperations(self, alpha_values, cv, train_data_features, train_data_label, test_data_features,
                                    test_data_label):
        try:
            ## performing elastic
            elastic = ElasticNetCV(alphas=alpha_values, cv=cv, normalize=True)
            elastic.fit(train_data_features, train_data_label)
            alpha = elastic.alpha_
            l1_ratio = elastic.l1_ratio
            ##building regression model using elastic
            elastic_lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            elastic_lr.fit(train_data_features, train_data_label)
            score = elastic_lr.score(test_data_features, test_data_label)

            return alpha, l1_ratio, score

        except Exception as e:
            raise Exception("Error occured while performing elastic operation", str(e))