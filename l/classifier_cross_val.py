import inspect
from itertools import product as iter_product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy.core.fromnumeric import product
import numpy as np
import category_encoders as ce
import statistics as stats
from functools import reduce, wraps


class ClassifierCrossVal:
    def __init__(self, X,y):
        self.X = X
        self.y = y
        self.encoders = None
        self.models = None
        self.scoring_methods = None
        self.k_folds = None

    def set_encoders(self, encoders):
        self.encoders = encoders

    def set_models(self, models):
        self.models = models

    def set_scoring_methods(self, methods):
        self.scoring_methods = methods

    def set_k_folds(self, folds):
        self.k_folds = folds

    target_mapping = {'LC': 0, 'NT': 1, 'EN': 2, 'VU': 3, 'CR': 4, 'DD': 5, 'CR (PE)': 6}

    #Will try all combinations of models and encoders
    def run(self):
        for encoder, model in iter_product(self.encoders, self.models):
            print("***************************************************************************")
            print("VALIDATING: ",encoder.__class__.__name__ + " + " + model.__class__.__name__ )
            print("***************************************************************************")
            ############### CROSS_VAL ###############
            confusion_matrixes = []
            scores = []
            
            folder = StratifiedKFold(n_splits=self.k_folds)
            for training_index, testing_index in folder.split(self.X,self.y):
                X_train, X_test = self.X.iloc[training_index], self.X.iloc[testing_index]
                y_train, y_test = self.y.iloc[training_index], self.y.iloc[testing_index]


                ############### ENCODING ###############
                X_train_encoded = None
                y_train_encoded = None
                X_test_encoded = None
                y_test_encoded = None
                if self.is_encoder_y_dependent(encoder):
                    X_train_encoded, y_train_encoded = self._encode_y_dependent(X_train,y_train, encoder)
                    X_test_encoded, y_test_encoded = self._encode_y_dependent(X_test,y_test, encoder)
                else:
                    X_train_encoded, y_train_encoded = self._encode_y_independent(X_train,y_train, encoder)
                    X_test_encoded, y_test_encoded = self._encode_y_independent(X_test,y_test, encoder)
                
                X_train_encoded = X_train_encoded.reindex(labels=X_test_encoded.columns,axis=1)
                X_test_encoded = X_test_encoded.reindex(labels=X_train_encoded.columns,axis=1)
                y_train_encoded = y_train_encoded.values.ravel() 
                y_test_encoded = y_test_encoded.values.ravel()

                ############### FIT AND SCORE ###############

                model.fit(X_train_encoded, y_train_encoded)
                y_pred = model.predict(X_test_encoded)
                confusion_matrixes.append(confusion_matrix(y_test_encoded, y_pred,normalize='true'))
                scoring_dict = {}
                for scoring_method in self.scoring_methods:
                    scoring_dict[scoring_method[0]] = scoring_method[1](y_test_encoded,y_pred)
                scores.append(scoring_dict)
            #Print:
            mean_conf_matrix = np.mean(confusion_matrixes, axis=0)
            mean_scores = self._avg_dicts(scores)
            print(mean_scores)
            to_display = ConfusionMatrixDisplay(mean_conf_matrix, display_labels=self.target_mapping.keys())
            to_display.plot(values_format='.1f')
            to_display.ax_.set_title(encoder.__class__.__name__ + " + " + model.__class__.__name__ + "\n" + str(mean_scores))
           
            

    def _add_dicts(self,dict_a, dict_b):
        sum_dict = {}
        for key in dict_a.keys():
            sum_dict[key] = dict_a[key] + dict_b[key]
        return sum_dict
    
    def _avg_dicts(self, list_of_dicts):
        total_dict = reduce(self._add_dicts, list_of_dicts)
        avg_dict = {}
        for key in total_dict.keys():
            avg_dict[key] = round(total_dict[key] / len(list_of_dicts),3)
        return avg_dict

    def is_encoder_y_dependent(self,encoder):
        signature = str(inspect.signature(encoder.transform))
        print(signature)
        if signature == "(X, y=None, override_return_df=False)":
            return True
        elif signature == "(X, override_return_df=False)":
            return False

            

    def _encode_y_dependent(self, X, y, encoder):
        ordinal_encoder =  ce.OrdinalEncoder(cols=['europeanRegionalRedListCategory'], mapping=[{'col': 'europeanRegionalRedListCategory',      'mapping': self.target_mapping }], return_df=True)
        y_encoded = ordinal_encoder.fit_transform(y)
        X_encoded = encoder.fit_transform(X, y_encoded)
        return (X_encoded, y_encoded)
    
    def _encode_y_independent(self, X, y, encoder):
        ordinal_encoder =  ce.OrdinalEncoder(cols=['europeanRegionalRedListCategory'], mapping=[{'col': 'europeanRegionalRedListCategory',      'mapping': self.target_mapping }], return_df=True)
        y_encoded = ordinal_encoder.fit_transform(y)
        X_encoded = encoder.fit_transform(X)
        return (X_encoded, y_encoded)

    