import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering with verification
def create_features(df):
    # Create new features
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
    
    # Keep original columns needed for modeling
    keep_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                    'Embarked', 'Deck', 'Title', 'FamilySize', 'IsAlone']
    
    return df[keep_columns + ['Survived']] if 'Survived' in df.columns else df[keep_columns]

# Apply feature engineering
train_processed = create_features(train)
test_processed = create_features(test)

# Verify columns exist
print("Train columns:", train_processed.columns.tolist())
print("Test columns:", test_processed.columns.tolist())

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Pclass', 'Sex', 'Embarked', 'Deck', 'Title', 'IsAlone'])
])

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Cross-validation
cv_scores = cross_val_score(model, 
                          train_processed.drop('Survived', axis=1), 
                          train_processed['Survived'], 
                          cv=5,
                          scoring='accuracy')
print(f"\nCross-validated accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Final training and prediction
model.fit(train_processed.drop('Survived', axis=1), train_processed['Survived'])
test_processed['Survived'] = model.predict(test_processed)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_processed['Survived']
})
submission.to_csv('submission.csv', index=False)