# California_housing

The file model_explorations.py trains an optimized SVM to predict housing prices for California, based on the 1990 California census data, as available at https://raw.githubusercontent.com/ageron/handson-ml/master/

It first trains a Random Forest to compute the relative importance of each attribute, dispenses with the least informative ones and trains a SVM using the remaining features. For both models either Grid Search or Randomized Search is used for cross-validated hyperparameter optimization

## Requirements

Anaconda, which includes all dependencies (numpy, scypy, pandas, sklearn)