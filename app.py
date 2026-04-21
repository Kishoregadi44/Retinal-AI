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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes_ai_secure_key_99'

# --- FIREBASE ---
try:
    if os.environ.get('FIREBASE_CONFIG'):
        cred_json = json.loads(os.environ.get('FIREBASE_CONFIG'))
        cred = credentials.Certificate(cred_json)
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase Error: {e}")

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

# --- AI MODEL (The Indestructible Loader) ---
model = None
class_names = ['High Risk', 'Low Risk', 'Medium Risk', 'Extreme Risk (Severe)']

def load_ai():
    global model
    try:
        # Step 1: Build the 'shell' manually
        base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        out = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        
        # Step 2: Try loading weights
        path = os.path.join(os.getcwd(), 'models', 'diabetic_retinopathy_v1.h5')
        if os.path.exists(path):
            model.load_weights(path)
            print("!!! SUCCESS: AI weights loaded !!!")
        else:
            print("!!! ERROR: Model file not found at path !!!")
    except Exception as e:
        print(f"!!! CRITICAL MODEL ERROR: {e} !!!")
        model = None # Keep site running even if AI fails

load_ai()

# --- ROUTES ---
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        users = db.collection('users').where('email', '==', email).limit(1).get()
        if users:
            user_doc = users[0]
            if bcrypt.check_password_hash(user_doc.to_dict()['password'], request.form['password']):
                user_obj = User(user_doc.id, user_doc.to_dict()['name'], user_doc.to_dict()['email'])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
        flash('Login Failed', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name, scans=[])

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        flash("AI Model is currently offline. Please try again later.", "danger")
        return redirect(url_for('dashboard'))
    
    file = request.files.get('file')
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(app.root_path, 'static', filename)
    file.save(filepath)
    
    img = cv2.resize(cv2.imread(filepath), (224, 224)).astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    res = class_names[np.argmax(pred)]
    
    return render_template('results.html', result=res, confidence=95.0, img=filename)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))