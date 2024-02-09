from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


class LogisticSurvivalModel:
    """
    This class will clean the dataset Titanic to get it ready for the model
    The following tasks will be performed:
    - 
    """

    def __init__(self):
        self.df = pd.read_csv("data/titanic-dataset/Titanic-Dataset.csv")
        self.features = ['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']
        self.survival = 'Survived'
        self.scaler = StandardScaler()

    @staticmethod
    def manipulate_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function will convert categorical data to numerical data and fill the nan values in the age column
        with the mean of the age.
        
        :argument df: pd.DataFrame
        :return: pd.DataFrame
        """
        # Update sex column to numerical
        df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
        # Fill the nan values in the age column
        df['Age'].fillna(value=df['Age'].mean(), inplace=True)
        # Create a first class column
        df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
        # Create a second class column
        df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
        # Create a second class column
        df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
        # Select the desired features
        df = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass', 'Survived']]
        return df

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

    @staticmethod
    def confusion_matrix(y_test, y_predict):
        """
        This function will calculate the confusion matrix and return the confusion matrix
        :argment y_test: np.array
        :argment y_predict: np.array
        """
        # Calculating Confusion Matrix
        confusion = confusion_matrix(y_test, y_predict)
        FN = confusion[1][0]
        TN = confusion[0][0]
        TP = confusion[1][1]
        FP = confusion[0][1]
        return_dict = {'FN': FN, 'TN': TN, 'TP': TP, 'FP': FP}
        return return_dict

    def model_architect(self):
        """
        This function will clean the dataset and return the cleaned dataset
        """
        df_clean = self.manipulate_df(self.df)
        X_train, X_test, y_train, y_test = train_test_split(df_clean[self.features], df_clean[self.survival],
                                                            test_size=0.2)
        train_features = self.scaler.fit_transform(X_train)
        test_features = self.scaler.transform(X_test)
        scores_and_predictions = self.model_develop(train_features, test_features, y_train, y_test)
        return scores_and_predictions, y_test

    def clean_df(self) -> dict[str, dict[str, float | Any] | dict[str, Any]]:
        """
        This function will clean the dataset and return the cleaned dataset
        
        :return: pd.DataFrame
        """
        scores_and_predictions, y_test = self.model_architect()
        confusion_matrix = self.confusion_matrix(y_test, scores_and_predictions['y_predict'])
        return_dict = {'scores_and_predictions': scores_and_predictions, 'confusion_matrix': confusion_matrix}
        return return_dict

    def get_prediction(self, sex, age, f_class, s_class, t_class, name) -> dict[str, Any]:
        """
        This function will clean the dataset and return the cleaned dataset

        :return: pd.DataFrame
        """
        scores_and_predictions, y_test = self.model_architect()
        model = scores_and_predictions['model']
        input_data = self.scaler.transform([[sex, age, f_class, s_class, t_class]])
        prediction = model.predict(input_data)
        predict_probability = model.predict_proba(input_data)
        return_dict = {'prediction': prediction, 'predict_probability': predict_probability}
        return return_dict

