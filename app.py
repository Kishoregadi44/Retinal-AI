import os
import uuid
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# --- Firebase Setup ---
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

# --- Original Model Loading ---
model = tf.keras.models.load_model('models/diabetic_retinopathy_v1.h5')
class_names = ['High Risk', 'Low Risk', 'Medium Risk', 'Extreme Risk (Severe)']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user_data = {
            'name': request.form['name'],
            'email': request.form['email'],
            'password': hashed_password
        }
        db.collection('users').add(user_data)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_query = db.collection('users').where('email', '==', email).limit(1).get()
        if user_query:
            user_doc = user_query[0]
            if bcrypt.check_password_hash(user_doc.to_dict()['password'], password):
                user_obj = User(user_doc.id, user_doc.to_dict()['name'], user_doc.to_dict()['email'])
                login_user(user_obj)
                return redirect(url_for('dashboard'))
        flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    scans = db.collection('scans').where('user_id', '==', current_user.id).stream()
    user_scans = [scan.to_dict() for scan in scans]
    return render_template('dashboard.html', name=current_user.name, scans=user_scans)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = str(uuid.uuid4()) + ".png"
        filepath = os.path.join('static', filename)
        file.save(filepath)
        
        # Original Image Processing
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        result = class_names[np.argmax(prediction)]
        
        db.collection('scans').add({
            'user_id': current_user.id,
            'result': result,
            'image_file': filename,
            'timestamp': datetime.now()
        })
        
        return render_template('results.html', result=result, img=filename)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)