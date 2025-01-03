import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Application settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
    
    # Model files
    MODEL_PATH = os.path.join(MODEL_FOLDER, 'xgboost_fraud_model.joblib')
    SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.joblib')
    
    # Model parameters
    MODEL_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'random_state': 42
    }
    
    # API settings
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size 