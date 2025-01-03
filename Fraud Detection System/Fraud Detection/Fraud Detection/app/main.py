import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shutil
import joblib

app = Flask(__name__)

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.pkl')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    METRICS_PATH = os.path.join(STATIC_FOLDER, 'metrics.json')

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        
        self.categorical_features = [
            'category',
            'state',
            'job'
        ]
        
        self.numerical_features = ['amt']

    def preprocess_data(self, df):
        """Enhanced preprocessing with feature engineering"""
        df_processed = df.copy()
        
        # Handle missing values
        for col in self.numerical_features:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        for col in self.categorical_features:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                df_processed[feature] = self.encoders[feature].fit_transform(df_processed[feature])
            else:
                df_processed[feature] = self.encoders[feature].transform(df_processed[feature])
        
        # Scale numerical features
        numerical_features = self.numerical_features
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])
        else:
            df_processed[numerical_features] = self.scaler.transform(df_processed[numerical_features])
        
        return df_processed

    def train(self):
        try:
            # Load datasets
            train_df = pd.read_csv(os.path.join(Config.UPLOAD_FOLDER, 'fraudTrain.csv'))
            test_df = pd.read_csv(os.path.join(Config.UPLOAD_FOLDER, 'fraudTest.csv'))
            
            # Select required columns
            columns_to_use = self.numerical_features + self.categorical_features + ['is_fraud']
            train_df = train_df[columns_to_use]
            test_df = test_df[columns_to_use]
            
            # Merge datasets
            df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            
            # Preprocess data with enhanced features
            processed_df = self.preprocess_data(df)
            
            # Prepare features and target
            features = self.numerical_features + self.categorical_features
            X = processed_df[features]
            y = df['is_fraud']
            
            # Use stratified split due to imbalanced data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model with optimized parameters
            self.model = xgb.XGBClassifier(
                max_depth=8,
                learning_rate=0.05,
                n_estimators=200,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                objective='binary:logistic',
                scale_pos_weight=10,  # Handle class imbalance
                random_state=42
            )
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Generate predictions
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_prob))
            }
            
            # Save metrics
            os.makedirs(Config.STATIC_FOLDER, exist_ok=True)
            with open(Config.METRICS_PATH, 'w') as f:
                json.dump(metrics, f)
            
            # Save model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise

    def predict(self, transaction_data):
        try:
            df = pd.DataFrame([transaction_data])
            
            # Preprocess the data
            processed_df = self.preprocess_data(df)
            
            # Check the columns of the processed DataFrame
            print("Processed DataFrame columns:", processed_df.columns.tolist())  # Debugging line
            
            features = self.numerical_features + self.categorical_features
            
            # Ensure that the features used for prediction match those used during training
            prediction_proba = self.model.predict_proba(processed_df[features])
            
            return prediction_proba[0][1]
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

    def save_model(self):
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(Config.ENCODERS_PATH, 'wb') as f:
            pickle.dump(self.encoders, f)

    def load_model(self):
        with open(Config.MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(Config.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(Config.ENCODERS_PATH, 'rb') as f:
            self.encoders = pickle.load(f)

    def get_unique_values(self):
        unique_values = {}
        for feature in self.categorical_features:
            unique_values[feature] = self.encoders[feature].classes_.tolist() if feature in self.encoders else []
        return unique_values

# Initialize model
fraud_model = FraudDetectionModel()

@app.route('/')
def home():
    metrics = {}
    unique_values = fraud_model.get_unique_values()  # Get unique values for dropdowns
    if os.path.exists(Config.METRICS_PATH):
        with open(Config.METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    return render_template('index.html', metrics=metrics, unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging line
        probability = fraud_model.predict(data)
        
        return jsonify({
            'is_fraudulent': bool(probability > 0.5),
            'confidence_score': float(probability)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create required directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.STATIC_FOLDER, exist_ok=True)
    
    print("\n----------------------------------------")
    print("🌐 Access the website at: http://127.0.0.1:5000")
    print("----------------------------------------\n")
    
    try:
        fraud_model.load_model()
        print("✅ Model loaded successfully")
    except:
        print("🔄 Training new model")
        fraud_model.train()
        print("✅ Model trained successfully")
    
    app.run(debug=True, host='127.0.0.1', port=5000) 