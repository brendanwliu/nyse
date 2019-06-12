from data_technicals_methods import *
from TimeS_stock_analysis import *

dataset_TI_df = get_technicals_indic(dataset_stock)
print(dataset_TI_df.head())
def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['close']
    X = data.iloc[:,2:]

    train_samples = int(X.shape[0] * 0.70)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)

# Get training and test data
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset_TI_df)
regressor = xgb.XGBRegressor(gamma=0,
                            n_estimators=75,
                            base_score=0.7,
                            colsample_bytree=1,
                            learning_rate=0.07,
                            max_depth = 5,
                            subsample = 0.8)
eval_metric = ["mae", "rmse"]
xgbModel = regressor.fit(X_train_FI,y_train_FI, \
                         eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], eval_metric = eval_metric)
eval_result = regressor.evals_result()
print(eval_result)
training_rounds = range(len(eval_result['validation_0']['mae']))
plt.scatter(x=training_rounds,y=eval_result['validation_0']['mae'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['mae'],label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
plt.title('Figure 6: Feature importance of the technical indicators.')
plt.show()
