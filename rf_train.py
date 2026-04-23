import numpy as np
import pandas as pd
import pickle
#tts
from sklearn.model_selection import train_test_split
#pp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#impute
from sklearn.impute import SimpleImputer
#col
from sklearn.compose import ColumnTransformer
#pipeline
from sklearn.pipeline import Pipeline
#rf
from sklearn.ensemble import RandomForestRegressor
#mse r2
from sklearn.metrics import mean_squared_error, r2_score

#load data
df = pd.read_csv('ModBSP.csv')

print(df)

#target and features

X = df.drop(['hsc_result','date'], axis=1)
y = df['hsc_result']
#col split
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

#preprocessing
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
#preprocessor
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
#rf_model
rf_model=RandomForestRegressor(
    n_estimators=200, 
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
#pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
#tts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train
rf_pipeline.fit(X_train, y_train)
#predict
y_pred = rf_pipeline.predict(X_test)
#evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

with open('student_rf_pipeline.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)


print("Model saved as student_rf_pipeline.pkl")