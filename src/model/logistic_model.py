from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogisticGeneralModel:
    """
    This class aims to perform a logistic regression model on a dataset
    """

    def __init__(self, target):
        self.df = pd.read_csv('data/the-ultimate-halloween-candy-power-ranking/candy-data.csv')
        self.target = target
        self.scaler = StandardScaler()

    def model_architect(self):
        """
        This function will clean the dataset and return the cleaned dataset
        """

        df_clean = self.df
        df_clean.drop(columns=['competitorname'], axis=1, inplace=True)
        df_features = df_clean.drop(columns=[self.target], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df_features, df_clean[self.target],
                                                            test_size=0.2)
        train_features = self.scaler.fit_transform(X_train)
        test_features = self.scaler.transform(X_test)
        scores_and_predictions = self.model_develop(train_features, test_features, y_train, y_test)
        return scores_and_predictions

    @staticmethod
    def model_develop(train_features, test_features, y_train, y_test):
        """
        This function will develop the model and return the confusion matrix
        :argment train_features: np.array
        :argment test_features: np.array
        :argment y_train: np.array
        :argment y_test: np.array
        """
        # Create and train the model
        model = LogisticRegression()
        model.fit(train_features, y_train)
        train_score = model.score(train_features, y_train)
        test_score = model.score(test_features, y_test)
        y_predict = model.predict(test_features)
        return_dict = {'train_score': train_score, 'test_score': test_score, 'y_predict': y_predict, 'model': model}
        return return_dict

    def get_prediction(self, feature_input) -> dict[str, Any]:
        """
        This function will clean the dataset and return the cleaned dataset

        :return: pd.DataFrame
        """
        scores_and_predictions = self.model_architect()
        model = scores_and_predictions['model']
        input_data = self.scaler.transform([feature_input])
        prediction = model.predict(input_data)
        predict_probability = model.predict_proba(input_data)
        return_dict = {'prediction': prediction, 'predict_probability': predict_probability}
        return return_dict

