Ames Housing Prices
==============================

This is my analysis of the Ames housing dataset from the Kaggle
competition. The analysis involves data cleaning and machine learning
analysis. Best performing models include linear regression with ridge
and lasso regulation. Decision tree, random forest and xgboost were
also tested. There are many limitations to the current analysis and
much room for improvement. Some possible improvements listed below.

Submitted rmse = 0.128

Goals for this project
------------

To explore different regression techniques and their application to a
complex housing dataset. Kaggle competitions afford a relatively clean
and straightforward dataset which are optimal for exploration of
different modeling techniques without the need for a lot of data
preparation.


Jupyter notebooks were used for initial data exploration, model comparisions, and visualizations. 



Visuals:
----------

Comparison of lasso predicted vs actual for test and validation set

![Lasso Regression](./lasso.png)

Comparison of the important lasso coefficients

![Lasso Coefficients](./lasso_coef.png)


Comparison of ridge predicted vs actual for test and validation set

![Ridge Regression](./ridge.png )

Comparison of the important ridge coefficients

![Ridge Coefficients](./ridge_coef.png)


Further work to improve model:
----------

- Needs better outlier elimination
- Look into better modeling of non-linear coefficients
- Possibly use interaction terms and higher order terms
- Test different feature combinations

Further work to improve algorithms:
----------
- scripts not well optimized and will not scale well without major modifications to code
- was built as a 'one off' project
- vast improvements can be made with sklearn pipelines
- clunky use of column names...
