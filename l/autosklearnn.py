from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import autosklearn
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy

if __name__ == "__main__":
        eu_red_list_without_unknown_population_trends = pd.read_csv("/Users/kristianjespersen/Documents/IKT/6_semester/ITMAL/itmal_4_redlist/datasets_ready/eu_red_list_without_unknown_population_trends.csv")

        X = eu_red_list_without_unknown_population_trends.iloc[:,:-1]
        y = eu_red_list_without_unknown_population_trends.iloc[:,-1]
        encoded_X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
        encoded_y = LabelEncoder().fit_transform(y)


        X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(encoded_X, encoded_y, random_state=1)
        automl = autosklearn.classification.AutoSklearnClassifier()
        print("Performing fit")
        automl.fit(X_train, y_train)
        y_hat = automl.predict(X_test)
        print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
        print(automl.show_models())
