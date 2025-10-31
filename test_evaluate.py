import unittest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from DropColumns import DropColumns
import mlflow.pyfunc

class EvaluateModel(unittest.TestCase): 
    model_uri =  'models/iris-model/Production'
    data_path = 'data/iris_data.csv'
    model_iris = None
    dat_iris = None

    def setUp(self):
        self.model_iris = mlflow.pyfunc.load_model(self.model_uri)
        self.data_iris = pd.read_csv(self.data_path)

    def evaluate(self):
        X_eval = self.data_iris 
        y_eval = self.data_iris['species']
        preds = self.model_iris.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        print(f"Accuracy: {acc}")
        return acc

    def test_evaluate_accuracy(self):
        eval_acc = self.evaluate()
        assert eval_acc > 0.9

if __name__ == '__main__':
    unittest.main()

