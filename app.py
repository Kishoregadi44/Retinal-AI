import os
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

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'retinal_ai_secret_key_2026'

# --- 1. FIREBASE CONFIGURATION ---
# This looks for your service account key to connect to your database
try:
    if os.environ.get('FIREBASE_CONFIG'):
        # For Render deployment
        cred_json = json.loads(os.environ.get('FIREBASE_CONFIG'))
        cred = credentials.Certificate(cred_json)
    else:
        # For Local VS Code development
        cred = credentials.Certificate("serviceAccountKey.json")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected successfully.")
except Exception as e:
    print(f"Error connecting to Firebase: {e}")

# --- 2. AUTHENTICATION SETUP ---
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

# --- 3. AI MODEL LOADING ---
# Path to your .h5 file inside the 'models' folder
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'diabetic_retinopathy_v1.h5')
try:
    # compile=False ensures the model loads even if TF versions vary
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("AI Model loaded successfully.")
except Exception as e:
    print(f"AI Model load error: {e}")
    model = None

class_names = ['High Risk', 'Low Risk', 'Medium Risk', 'Extreme Risk (Severe)']

# --- 4. ROUTES ---

@app.route('/')
def home():
    # If not logged in, go to login page
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        # Check if user exists
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
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users_ref = db.collection('users').where('email', '==', email).limit(1).get()
        if users_ref:
            user_doc = users_ref[0]
            user_data = user_doc.to_dict()
            if bcrypt.check_password_hash(user_data['password'], password):
                user_obj = User(user_doc.id, user_data['name'], user_data['email'])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
        
        flash('Login Unsuccessful. Check email and password.', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch scan history for the specific user
    try:
        scans_ref = db.collection('scans').where('user_id', '==', current_user.id)\
                      .order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        user_scans = [s.to_dict() for s in scans_ref]
    except Exception as e:
        print(f"Error fetching scans: {e}")
        user_scans = []
    
    return render_template('dashboard.html', name=current_user.name, scans=user_scans)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))

    if file and model:
        # Save file securely
        filename = f"{uuid.uuid4().hex}.png"
        static_path = os.path.join(app.root_path, 'static')
        if not os.path.exists(static_path):
            os.makedirs(static_path)
        
        filepath = os.path.join(static_path, filename)
        file.save(filepath)
        
        # Image Preprocessing
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224)).astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        result_idx = np.argmax(prediction)
        final_result = class_names[result_idx]
        confidence = round(float(np.max(prediction)) * 100, 2)

        # Save record to Firestore
        db.collection('scans').add({
            'image_file': filename,
            'result': final_result,
            'confidence': confidence,
            'user_id': current_user.id,
            'timestamp': datetime.now()
        })
        
        return render_template('results.html', 
                               result=final_result, 
                               confidence=confidence, 
                               img=filename)
    
    flash('AI Model is not ready. Please wait.', 'warning')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Using port 5001 to avoid conflicts
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)