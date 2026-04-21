import os
# Force legacy Keras behavior immediately
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

# Import TensorFlow
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes_ai_secure_key_99'

# --- FIREBASE INITIALIZATION ---
if os.environ.get('FIREBASE_CONFIG'):
    cred_json = json.loads(os.environ.get('FIREBASE_CONFIG'))
    cred = credentials.Certificate(cred_json)
else:
    # Local fallback
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
    except:
        cred = None

if cred and not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- USER CLASS ---
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

# --- AI SETUP: ROBUST LOADER ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / 'models' / 'diabetic_retinopathy_v1.h5')

def load_retinal_model():
    # Clear session to prevent memory overhead
    tf.keras.backend.clear_session()
    
    # We use compile=False to ignore the metadata that causes the 'InputLayer' error
    # This bypasses the optimizer and training configuration, loading only weights and layers
    try:
        # Try loading via the standard TF Keras loader with strict safety off
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception:
        # Final fallback: Use the Keras 2 legacy loader specifically
        import tf_keras
        return tf_keras.models.load_model(MODEL_PATH, compile=False)

model = load_retinal_model()
class_names = ['High Risk', 'Low Risk', 'Medium Risk', 'Extreme Risk (Severe)']

# --- ROUTES ---

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        user_check = db.collection('users').where('email', '==', email).get()
        if len(user_check) > 0:
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        hashed_pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user_ref = db.collection('users').document()
        user_ref.set({
            'name': request.form['name'],
            'email': email,
            'password': hashed_pw,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

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
        flash('Login failed. Check email and password.', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        scans_ref = db.collection('scans').where('user_id', '==', current_user.id)\
                      .order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        user_scans = []
        for s in scans_ref:
            data = s.to_dict()
            user_scans.append(data)
            
        return render_template('dashboard.html', name=current_user.name, scans=user_scans)
    except Exception as e:
        print(f"Dashboard Query Issue: {e}")
        return render_template('dashboard.html', name=current_user.name, scans=[])

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash("No file selected", "danger")
        return redirect(url_for('dashboard'))
    
    filename = f"{uuid.uuid4().hex}.png"
    # Ensure static folder exists
    static_path = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_path):
        os.makedirs(static_path)
        
    filepath = os.path.join(static_path, filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    final_result = class_names[np.argmax(prediction)]
    confidence_score = round(float(np.max(prediction)) * 100, 2)

    # Note: Recommendations based on result
    if "Low" in final_result:
        diet = "Maintain a balanced diet rich in leafy greens."
        activity = "30 mins of moderate cardio 5 days a week."
    elif "Medium" in final_result:
        diet = "Reduce sugar intake; focus on low-glycemic foods."
        activity = "Daily brisk walking."
    else: 
        diet = "Strict diabetic diet; consult a specialist."
        activity = "Light movement only; follow medical supervision."

    scan_data = {
        'image_file': filename, 
        'result': final_result, 
        'confidence': confidence_score,
        'user_id': current_user.id,
        'diet_note': diet,
        'activity_note': activity,
        'timestamp': datetime.now()
    }
    db.collection('scans').add(scan_data)
    
    return render_template('results.html', result=final_result, confidence=confidence_score, 
                           img=filename, diet=diet, activity=activity)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)