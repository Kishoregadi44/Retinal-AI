import os
# This must be the very first line to tell TensorFlow to act like the old version
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import uuid
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes_ai_secure_key_99'

# --- FIREBASE ---
if os.environ.get('FIREBASE_CONFIG'):
    cred_json = json.loads(os.environ.get('FIREBASE_CONFIG'))
    cred = credentials.Certificate(cred_json)
else:
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_id, name, email):
        self.id = user_id
        self.name = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user_doc = db.collection('users').document(user_id).get()
    if user_doc.exists:
        data = user_doc.to_dict()
        return User(user_id, data['name'], data['email'])
    return None

# --- AI MODEL LOADING ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / 'models' / 'diabetic_retinopathy_v1.h5')

# We use compile=False to avoid the "InputLayer" metadata error
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully with TF 2.15")
except Exception as e:
    print(f"Loading failed: {e}")
    model = None

class_names = ['High Risk', 'Low Risk', 'Medium Risk', 'Extreme Risk (Severe)']

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        users_ref = db.collection('users').where('email', '==', email).limit(1).get()
        if users_ref:
            user_doc = users_ref[0]
            user_data = user_doc.to_dict()
            if bcrypt.check_password_hash(user_data['password'], request.form['password']):
                user_obj = User(user_doc.id, user_data['name'], user_data['email'])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
        flash('Login failed.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        db.collection('users').add({
            'name': request.form['name'],
            'email': request.form['email'],
            'password': hashed_pw,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    scans_ref = db.collection('scans').where('user_id', '==', current_user.id).stream()
    user_scans = [s.to_dict() for s in scans_ref]
    return render_template('dashboard.html', name=current_user.name, scans=user_scans)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    file = request.files.get('file')
    if not file: return redirect(url_for('dashboard'))
    
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(app.root_path, 'static', filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    final_result = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    db.collection('scans').add({
        'image_file': filename, 
        'result': final_result, 
        'confidence': confidence,
        'user_id': current_user.id,
        'timestamp': datetime.now()
    })
    
    return render_template('results.html', result=final_result, confidence=confidence, img=filename)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)