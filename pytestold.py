import unittest
import joblib
from DropColumns import DropColumns
import pandas as pd

class TestModel(unittest.TestCase):
    top_model = 'models/model.joblib'
    samples = 'data/iris_data.csv'
    model_dt = None

    def setUp(self):
            self.model_dt = joblib.load(self.top_model)

    def test_sample1(self):
        sample_data = pd.read_csv(self.samples)
        print(sample_data.head())
        
        prediction = self.model_dt.predict(sample_data.head(1))
        print("Prediction:", prediction)
        self.assertEqual(prediction,'setosa','Prediction is wrong')

if __name__ == '__main__':
    unittest.main()

