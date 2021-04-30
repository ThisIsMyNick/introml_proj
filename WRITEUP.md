# Iowa House Price

## Intro

I acquired data from a [Kaggle competition](https://www.kaggle.com/c/iowa-house-price-prediction).
This data included 80 columns, with the target being the sale price of a house.

I removed non-numeric columns and focused on numerical data.

Here you can see information some of the columns in the dataset.
![Data subsection](/imgs/data_describe.png)

The most important column to understand is the sale price, which we are trying to predict.

![Sale Price Line Plot](/imgs/saleprices_lineplot.png)

![Sale Price Box Plot](/imgs/saleprices_boxplot.png)

We can see from these graphs that the majority of prices are centered around the mean of 185k, and there are some significant outliers at higher price points.

## Logistic/Ridge regression

I tested several parameters for logistic and ridge regression.

```
reg_pipe_standard = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(max_iter=5000))
reg_pipe_minmax = make_pipeline(preprocessing.MinMaxScaler(), LogisticRegression(max_iter=5000))

reg_pipe_standard.fit(X_train, y_train)
reg_pipe_minmax.fit(X_train, y_train)

ridge_reg_std_1 = make_pipeline(preprocessing.StandardScaler(), Ridge(alpha=0.01))
ridge_reg_std_1.fit(X_train, y_train)

ridge_reg_minmax_1 = make_pipeline(preprocessing.MinMaxScaler(), Ridge(alpha=0.01))
ridge_reg_minmax_1.fit(X_train, y_train)

ridge_reg_std_2 = make_pipeline(preprocessing.StandardScaler(), Ridge(alpha=0.1))
ridge_reg_std_2.fit(X_train, y_train)

ridge_reg_minmax_2 = make_pipeline(preprocessing.MinMaxScaler(), Ridge(alpha=0.1))
ridge_reg_minmax_2.fit(X_train, y_train)
```

I'm testing StandardScaler() vs. MinMaxScaler() as ways to preprocess values, and two alpha values for ridge regression.

I use these models to calculate the root mean squared error of train/test sets, displayed below.

![Logistic Regression Results](/imgs/logreg_results.png)

We can see from this table that StandardScaler performed better than MinMaxScaler.
The difference is great with logistic regression, but becomes very small with ridge regression.

The best performer from this set of models is ridge regression with alpha=0.01, with a RMSE of 32.8k.

## SVM

I used grid search to find the best hyperparameters for an SVM model as such

```
svm_clf = GridSearchCV(SVC(), {"kernel": ("linear", "poly", "rbf"), "C": [1,5,10]}, scoring="neg_root_mean_squared_error")

svc_pipe = make_pipeline(preprocessing.StandardScaler(), svm_clf)
svc_pipe.fit(X_train, y_train)
```

Here I test three kernels, 3 C's, and use root mean squared error as the scoring function (it is negated because higher is better for the scoring function).

The results:

![SVM Results](/imgs/svm_results.png)

We can see that the linear kernel outperforms the other two, and has consistent error throughout different C's.
However, a higher C value helped the rbf and polynomial kernels.

The best model here is linear kernel with C=1.

## Neural Network

For the neural network, I used a multi-layer perceptron with two hidden layers with 6 nodes each.
I really struggled to find the optimal configuration of the neural network without reducing the number of layers,
which makes me think this data is not suited particularly well to a complex neural network model and perhaps simpler models are sufficient.

![Neural network results](/imgs/nn_results.png)

We can see that the best performer here is the MLP with alpha=0.01.

## Conclusions

The best model was logistic regression (standard scaler, alpha=0.01) with a MRSE of 32k.

The average house price is 185k, and the standard deviation is 83k. An error of 32k doesn't look too bad in this context.

If I had more time, I would improve this model by taking into account the numerous categorical columns that I had to drop.
This includes information such as build materials, garage type, and quality values for basement, garage, and kitchen.
This data would no doubt help more accurately shape predictions.
