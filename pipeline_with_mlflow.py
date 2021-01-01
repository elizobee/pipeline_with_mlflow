import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

#read in the data from UCI archive of Cleveland Clinic data
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
df = pd.read_csv(URL, header=None, names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'])

#define the target variable
df['target']=np.where(df['num'] > 0,1,0)

#review the data
df.head()

#create train, test, and validation data
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'Train Data')
print(len(val), 'Validation Data')
print(len(test), 'Test Data')

#import the custom transformer class
from custom_class import NewFeatureTransformer

#Define the Pipeline
numeric_features = ['age','trestbps','chol','thalach','oldpeak']
numeric_transformer = Pipeline(steps=[
     ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler())])

categorical_features = [ 'cp','restecg','ca','thal','slope']
categorical_transformer = Pipeline(steps=[ 
     ('imputer', SimpleImputer(strategy='constant',fill_value=0)),
     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

binary_features = [ 'sex','fbs','exang']
binary_transformer = Pipeline(steps=[
     ('imputer',SimpleImputer(strategy='constant',fill_value=0))])

new_features_input =  [ 'thalach','trestbps']
new_transformer = Pipeline(steps=[
     ('new', NewFeatureTransformer())])

preprocessor = ColumnTransformer(
     transformers=[
          ('num', numeric_transformer, numeric_features),
          ('cat', categorical_transformer, categorical_features),
          ('binary', binary_transformer, binary_features),
          ('new', new_transformer, new_features_input)])

# Now join together the preprocessing with the classifier.
clf = Pipeline(steps=[('preprocessor', preprocessor),
     ('classifier', LogisticRegression())], verbose=True)

#fit the pipeline
clf.fit(train, train['target'].values)

#create predictions for validation data
y_pred = clf.predict(val)

class ModelOut (mlflow.pyfunc.PythonModel):
     def __init__(self, model):
          self.model = model
     def predict (self, context, model_input):
          model_input.columns= map(str.lower,model_input.columns)
          return self.model.predict_proba(model_input)[:,1]
          
mlflow_conda={'channels': ['defaults'],
     'name':'conda',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow==1.11.0','scikit-learn==0.23.2','cloudpickle==1.5.0','pandas==1.1.1','numpy==1.19.1']}]}

with mlflow.start_run():
     #log metrics
     mlflow.log_metric("accuracy", accuracy_score( val['target'].values, y_pred))
     mlflow.log_metric("precison", precision_score( val['target'].values, y_pred))
     mlflow.log_metric("recall", recall_score( val['target'].values, y_pred))
     # log model
     mlflow.pyfunc.log_model(   artifact_path="model",
          python_model=ModelOut(model=clf,),
          code_path=['custom_class.py'],
          conda_env=mlflow_conda)
     signature = infer_signature(val, y_pred)
     #print out the active run ID
     run = mlflow.active_run()
     print("Active run_id: {}".format(run.info.run_id))