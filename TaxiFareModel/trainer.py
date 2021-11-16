from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        
        return None

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        return None

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred=y_pred, y_true=y_test)



if __name__ == "__main__":
    
    df = get_data()
    df = clean_data(df)
    
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # build pipeline
    trainer = Trainer(X_train,y_train)

    # train the pipeline
    trainer.run()
    print(trainer.evaluate(X_val,y_val))
