# This script trains an optimized SVM to predict housing prices for California, based on the 1990 California census data, as available at https://raw.githubusercontent.com/ageron/handson-ml/master/
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from categorical_encoder import * # Where the CategoricalEncoder class is
from scipy.stats import randint, expon
import numpy as np
import pandas as pd
import os
CSV_PATH = "datasets/housing.csv"
def load_housing_data(csv_path=CSV_PATH):
    return pd.read_csv(csv_path)
housing = load_housing_data()

# A new attribute is created to perform stratified sampling, given that median income is very important and we want the test set to be representative of various income strata
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# The distribution of the new attribute's various strata is compared between the whole dataset and the stratified test set
#print (housing["income_cat"].value_counts()/len(housing))
#print (strat_test_set["income_cat"].value_counts()/len(strat_test_set))

# The added attribute is discarded from the sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Features and Label sets are computed
strat_train_set_features = strat_train_set.drop("median_house_value", axis=1)
strat_train_set_labels = strat_train_set["median_house_value"].copy()
strat_test_set_features = strat_test_set.drop("median_house_value", axis=1)
strat_test_set_labels = strat_test_set["median_house_value"].copy()

# Custom transformer for combining features into novel ones
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        bedrooms_per_room =  X[:, bedrooms_ix]/X[:, rooms_ix]
        if self.add_bedrooms_per_room:
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

# Custom transformer for selecting which attributes will go down the pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_attributes):
        self.selected_attributes = selected_attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.selected_attributes].values

# Lists of both numerical and categorical attributes are computed
num_attributes_list = list(strat_train_set_features.drop("ocean_proximity", axis=1))
cat_attributes_list  = ["ocean_proximity"]

# Data prep transformation sequences are abstracted as pipelines
num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attributes_list)),
    ("imputer", Imputer(strategy='median')),
    ("attributes_adder", CombinedAttributesAdder()),
    ("scaler", StandardScaler())
    ])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attributes_list)),
    ("binary_encoding", CategoricalEncoder(encoding="onehot-dense")) # Encoded for array output
    ])

# A unified pipeline is constructed
unified_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
    ])

# Both train and test feature sets are run through the pipeline
prepared_train_set_features = unified_pipeline.fit_transform(strat_train_set_features)
prepared_test_set_features = unified_pipeline.transform(strat_test_set_features)

# A performant random forest is trained using cross-validation
param_grid = [
    {"n_estimators": [3,10,30], "max_features":[4,6,8],},
    {"n_estimators": [3,10], "max_features":[2,4,6], "bootstrap":[False]},
    ]
rnd_forest = RandomForestRegressor(random_state=42)
rnd_forest_search = GridSearchCV(rnd_forest, param_grid, cv=3, scoring='neg_mean_squared_error', return_train_score=True, verbose=4)
rnd_forest_search.fit(prepared_train_set_features, strat_train_set_labels)
forest_model = rnd_forest_search.best_estimator_
#joblib.dump(forest_model, "forest_model.pkl")
#forest_model = joblib.load("forest_model.pkl")

# Prints out each attribute name along its relative importance for determining the target value
cat_attributes_names = list(cat_pipeline.named_steps["binary_encoding"].categories_[0])
attribute_names = num_attributes_list+["rooms_per_household", "population_per_household", "bedrooms_per_room"]+cat_attributes_names
print (sorted(zip(forest_model.feature_importances_, attribute_names), reverse=True))

# The threshold is to be set manually, according to the relative importances of attributes and their distribution
threshold = 0.015
indices_to_be_deleted = []
for attribute_index in range(len(forest_model.feature_importances_)):
    if forest_model.feature_importances_[attribute_index]< threshold:
        indices_to_be_deleted.append(attribute_index)

# Uninformative attributes are dropped from the traning set before feeding it to the next model
prepared_train_set_features = np.delete(prepared_train_set_features, indices_to_be_deleted,axis=1)

# Cross-validation is used to train and test various SVMs using Grid Search and Randomized Search
svm = SVR(kernel="linear")

# Uncomment to perform Grid Search hyperparameter optimization instead of Randomized Search
'''
param_grid = [
    {
    "kernel":["linear"],
    "C":[30000,120000,],
    },
    {
    "kernel":["rbf"],
    "C":[40000,150000,],
    "gamma":[0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    },
]
search = GridSearchCV(svm, param_grid, cv=3, scoring='neg_mean_squared_error', return_train_score=True, verbose=True)
'''
param_dist = {
    'C': randint(low=140000, high=160000),
    'gamma': [0.26],#expon(scale=1.0),
    'kernel': ["rbf"],
}
search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=4, random_state=42)

# We fit the data into the estimator and print out the scores along with the parameters used
search.fit(prepared_train_set_features, strat_train_set_labels)
print (search.best_params_)
cvres = search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print (np.sqrt(-mean_score), params)

# The best model is used against the test set and the RMSE is computed
final_model = search.best_estimator_
prepared_test_set_features = np.delete(prepared_test_set_features, indices_to_be_deleted,axis=1)
predictions = final_model.predict(prepared_test_set_features)
RMSE = np.sqrt(mean_squared_error(strat_test_set_labels,predictions))
print (RMSE)
print(type(predictions))

#joblib.dump(final_model, "model.pkl")
#final_model = joblib.load("model.pkl")