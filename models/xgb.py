from xgboost import XGBRegressor
import numpy as np
from sklearn.model_selection import train_test_split

class XGBModel:
    def __init__ (self, X_location, y_location, test_size=0.2, val_size=0.1):
        self.X_location = X_location
        self.y_location = y_location
        self.X = np.load(X_location)
        self.y = np.load(y_location)

        # First split off the test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

        self.val_size = val_size
        if val_size and val_size > 0:
            # Adjust val_size to be a fraction of the remaining training set to keep it consistent
            # e.g. if test_size = 0.2 and val_size = 0.1, the relative val_size should be 0.1 / (1 - 0.2) = 0.125
            adjusted_val_size = val_size / (1 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=adjusted_val_size, random_state=42
            )
        else:
            self.X_val, self.y_val = None, None

    def set_model(self, params=None):
        if params == None:
            self.model = XGBRegressor()
        else:
            self.model = XGBRegressor(**params)

    def view_data(self):
        print("X.shape: ", self.X.shape)
        print("y.shape: ", self.y.shape)
        print("X_train.shape: ", self.X_train.shape)
        if self.X_val is not None:
            print("X_val.shape: ", self.X_val.shape)
        print("X_test.shape: ", self.X_test.shape)
        print("y_train.shape: ", self.y_train.shape)
        if self.y_val is not None:
            print("y_val.shape: ", self.y_val.shape)
        print("y_test.shape: ", self.y_test.shape)

    def train(self, **fit_params):
        if self.X_val is not None and self.y_val is not None:
            # Provide eval_set for monitoring and/or early stopping
            eval_set = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
            self.model.fit(self.X_train, self.y_train, eval_set=eval_set, **fit_params)
        else:
            self.model.fit(self.X_train, self.y_train, **fit_params)

    def evaluate(self):
        if self.X_val is not None and self.y_val is not None:
            y_val_pred = self.model.predict(self.X_val)
            val_mse = np.mean((y_val_pred - self.y_val)**2)
            print("Validation Mean Squared Error: ", val_mse)

        y_pred = self.model.predict(self.X_test)
        mse = np.mean((y_pred - self.y_test)**2)
        print("Test Mean Squared Error: ", mse)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, location):
        self.model.save_model(location)

    def load(self, location):
        self.model.load_model(location)
