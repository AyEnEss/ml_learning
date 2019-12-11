'''A collection of common functions that are usefull for ML algorithms'''

# ---COMMON IMPORTS---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ---FUNCTIONS---
def plot_learning_curves(model, X, y):
    '''
    Given a model, this function will plot the learning curves for that model
    parameters:
        model: an ml model to analyze
        X: should be a df or an array of the features
        y: series or 1d array. Should be the labels of the data.

    returns:
        A plot for the learning curves for the given model
    '''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [],[]

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')
    plt.legend(loc='upper right', fontsize=14)
    plt.xlabel('Training set size',fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.axis = ([0, 80, 0, 3])

    plt.show()


def test_run():
    '''A function to test and develop the functions in this file'''

    #initiate random points
    from sklearn.linear_model import LinearRegression

    m = 100
    X = 6 * np.random.rand(m, 1) -3
    y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)

    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)


if __name__ == '__main__':
    test_run()