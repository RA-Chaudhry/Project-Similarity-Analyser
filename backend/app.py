import os
import sys
import subprocess
import importlib.util
import faiss
import pickle
import time
import threading
from transformers import T5Tokenizer, T5EncoderModel

VENV_DIR = "venv"

def in_venv():
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

def create_venv():
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

def relaunch_with_venv():
    if os.name == "nt":
        python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(VENV_DIR, "bin", "python")
    print(f"Re-launching using {python_exe} ...")
    os.execv(python_exe, [python_exe] + sys.argv)

if not in_venv():
    if not os.path.isdir(VENV_DIR):
        create_venv()
    relaunch_with_venv()

# --- Auto-install dependencies if needed ---
REQUIREMENTS_FILE = "requirements.txt"
REQUIRED_PACKAGES = [
    "flask",
    "flask_sqlalchemy",
    "flask_cors",
    "numpy",
    "sentence-transformers",
    "scikit-learn",
    "redis",
    "werkzeug",
    "pyjwt"
]

def ensure_requirements_file():
    if not os.path.exists(REQUIREMENTS_FILE):
        with open(REQUIREMENTS_FILE, "w") as f:
            for pkg in REQUIRED_PACKAGES:
                f.write(pkg + "\n")

def install_missing_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            # Try to import the package (handle dash/underscore differences)
            importlib.import_module(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

ensure_requirements_file()
install_missing_packages()

from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import logging
from werkzeug.security import check_password_hash, generate_password_hash
import json
import jwt
import datetime
from functools import wraps

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///projects.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS for all origins
CORS(app, supports_credentials=True, resources={r"*": {"origins": "*"}})

# Database setup
db = SQLAlchemy(app)

# Redis setup
try:
    cache = redis.Redis(host='localhost', port=6379, db=0)
    cache.ping()
    logger.info("Successfully connected to Redis.")
except redis.ConnectionError as e:
    logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
    cache = None

# Load AI models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load T5 model and tokenizer
print("Loading T5 model...")
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5EncoderModel.from_pretrained('t5-small')

# Function to generate T5 features
def generate_t5_features(texts):
    inputs = t5_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = t5_model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.detach().numpy()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(50), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    results = db.Column(db.JSON, nullable=True)  # New field for storing similarity results
    user = db.relationship('User', backref=db.backref('histories', lazy=True))

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    history_id = db.Column(db.Integer, db.ForeignKey('history.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True))
    history = db.relationship('History', backref=db.backref('feedbacks', lazy=True))

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<Contact {self.name}>'


# Helper: Token Required Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.secret_key, algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                raise Exception('User not found')
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Preprocess and similarity logic (same as before)
def preprocess_text(text):
    return text.lower().strip()

# Update compute_project_embedding to use both SBERT and T5

def compute_project_embedding(title, summary):
    text = preprocess_text(f"{title} {summary} {summary}")
    sbert_features = sbert_model.encode([text])
    t5_features = generate_t5_features([text])
    features = np.concatenate([sbert_features, t5_features], axis=1)
    return features[0]

# --- FAISS Integration for Fast Similarity Search ---
FAISS_INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "project_embeddings.pkl"
ID_MAP_FILE = "project_id_map.pkl"

# Build or load FAISS index and project id map
faiss_index = None
project_id_map = []  # index -> project_id
project_embeddings = None

def build_faiss_index():
    global faiss_index, project_id_map, project_embeddings
    projects = Project.query.all()
    if not projects:
        faiss_index = None
        project_id_map = []
        project_embeddings = None
        return
    embeddings = []
    id_map = []
    for proj in projects:
        emb = compute_project_embedding(proj.title, proj.summary)
        embeddings.append(emb)
        id_map.append(proj.id)
    embeddings = np.stack(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss_index = index
    project_id_map = id_map
    project_embeddings = embeddings
    # Save to disk
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(ID_MAP_FILE, 'wb') as f:
        pickle.dump(id_map, f)

def load_faiss_index():
    global faiss_index, project_id_map, project_embeddings
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        with open(EMBEDDINGS_FILE, 'rb') as f:
            project_embeddings = pickle.load(f)
        with open(ID_MAP_FILE, 'rb') as f:
            project_id_map = pickle.load(f)
    except Exception:
        build_faiss_index()

# Call this on app startup
with app.app_context():
    db.create_all()
    load_faiss_index()

# Call this after any project add/update/delete
# build_faiss_index()

# Refactor calculate_similarity to use FAISS

def calculate_similarity(title, summary):
    t0 = time.time()
    cache_key = f"similarity_{title}_{summary}"
    if cache and cache.ping():
        cached_result = cache.get(cache_key)
        if cached_result:
            cached_data = json.loads(cached_result.decode('utf-8'))
            top_two = []
            for item in cached_data:
                proj = Project.query.get(item['id'])
                if proj:
                    top_two.append((proj, item['similarity']))
            return top_two

    # Compute embedding for user input
    user_emb = compute_project_embedding(title, summary).astype('float32')
    if faiss_index is None or len(project_id_map) == 0:
        return []
    D, I = faiss_index.search(user_emb.reshape(1, -1), 2)  # top 2
    t1 = time.time()
    print(f"SBERT encode: {t1 - t0:.3f}s, FAISS search: {D[0][0]:.3f}s, Total: {t1 - t0:.3f}s")
    top_two = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(project_id_map):
            continue
        proj_id = project_id_map[idx]
        proj = Project.query.get(proj_id)
        if proj:
            # Convert L2 distance to cosine similarity percentage (approximate)
            # Here, we use: similarity = 1 / (1 + dist) * 100
            similarity = 1 / (1 + dist) * 100
            top_two.append((proj, similarity))

    # After finding top_two, filter out duplicates by title
    unique_titles = set()
    unique_top_two = []
    for proj, similarity in top_two:
        if proj.title not in unique_titles:
            unique_top_two.append((proj, similarity))
            unique_titles.add(proj.title)
        if len(unique_top_two) == 2:
            break
    top_two = unique_top_two

    if cache and cache.ping():
        cache_data = [{'id': proj.id, 'similarity': float(similarity)} for proj, similarity in top_two]
        cache.set(cache_key, json.dumps(cache_data), ex=3600)
    return top_two

index_rebuild_lock = threading.Lock()
index_rebuild_running = False

def rebuild_faiss_index_async():
    def wrapped():
        global index_rebuild_running
        with index_rebuild_lock:
            if index_rebuild_running:
                print("FAISS index rebuild already running, skipping this request.")
                return  # Skip if already running
            index_rebuild_running = True
        try:
            with app.app_context():
                build_faiss_index()
        finally:
            with index_rebuild_lock:
                index_rebuild_running = False
    thread = threading.Thread(target=wrapped)
    thread.start()

# Routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API. Use /login to authenticate."})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '').strip()

    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()

    if user and user.check_password(password):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, app.secret_key, algorithm='HS256')

        resp = make_response(jsonify({"message": "Login successful"}))

        # 🔥 Fix: Make sure cookie is actually set
        resp.set_cookie(
            'token',
            token,
            httponly=True,
            secure=False,  # Set True for HTTPS
            samesite='Lax'  # Change from 'Strict' to 'Lax'
        )

        return resp

    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    # Check for missing fields
    if not name or not email or not username or not password:
        return jsonify({"error": "All fields  are required."}), 400

    # Check for duplicate email or username
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email is already registered."}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username is already taken."}), 400

    # Create and save user
    new_user = User(name=name, email=email, username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Signup successful"}), 200


@app.route('/logout')
def logout():
    resp = make_response(jsonify({"message": "Logged out"}))
    resp.set_cookie('token', '', expires=0)
    return resp

@app.route('/user', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({
        "status": "success",
        "user": {
            "name": current_user.name,
            "email": current_user.email,
            "username": current_user.username
        }
    }), 200


@app.route('/user/updateEmail', methods=['POST'])
@token_required
def update_email(current_user):
    data = request.get_json()
    current_password = data.get('password', '').strip()
    new_email = data.get('newEmail', '').strip()

    # Check if the current password is correct
    if not current_user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 400

    # Validate the new email
    if not new_email:
        return jsonify({"error": "New email is required"}), 400

    # Check if the new email is already in use
    existing_user = User.query.filter_by(email=new_email).first()
    if existing_user:
        return jsonify({"error": "Email is already in use"}), 400

    # Update the user's email
    current_user.email = new_email
    db.session.commit()

    return jsonify({"message": "Email updated successfully"}), 200

@app.route('/user/updatePassword', methods=['POST'])
@token_required
def update_password(current_user):
    app.logger.info("Update password route hit")
    data = request.get_json()
    current_password = data.get('password', '').strip()
    new_password = data.get('newPassword', '').strip()

    # Check if the current password is correct
    if not current_user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 400

    # Validate the new password
    if not new_password or len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters long"}), 400

    # Update the user's password
    current_user.set_password(new_password)
    db.session.commit()

    return jsonify({"message": "Password updated successfully"}), 200

@app.route('/index', methods=['POST'])
@token_required
def index(current_user):
    request_start = time.time()
    data = request.get_json()
    title = data.get('title', '').strip()
    summary = data.get('summary', '').strip()
    domain = data.get('domain', '').strip()
    history_id = data.get('historyId', None)  # Get the historyId from the request

    # Calculate similarity between the new project and existing projects
    top_matches = calculate_similarity(title, summary)
    route_end = time.time()
    print(f"/index route after similarity: {route_end - request_start:.3f}s")

    if history_id:
        # If historyId is provided, update the existing history
        existing_history = History.query.filter_by(id=history_id, user_id=current_user.id).first()
        if existing_history:
            # Update the existing history entry with new title, summary, and domain
            existing_history.title = title
            existing_history.summary = summary
            existing_history.domain = domain
            # Update the results with the new similarity check
            existing_history.results = [{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': float(similarity)
            } for proj, similarity in top_matches]

            db.session.commit()
            response = jsonify({
                "historyId": existing_history.id,  # Return the historyId of the updated history
                "matches": [{
                    'id': proj.id,
                    'title': proj.title,
                    'summary': proj.summary,
                    'domain': proj.domain,
                    'similarity': float(similarity)
                } for proj, similarity in top_matches],
                "message": "Similarity Checked"
            })
            request_end = time.time()
            print(f"/index route total time (update): {request_end - request_start:.3f}s")
            rebuild_faiss_index_async()  # Rebuild index in background
            return response
        else:
            return jsonify({"error": "History not found"}), 404  # If no history entry is found
    else:
        # If no historyId is provided, create a new history entry
        new_history = History(
            user_id=current_user.id,
            title=title,
            summary=summary,
            domain=domain,
            results=[{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': float(similarity)
            } for proj, similarity in top_matches]
        )
        db.session.add(new_history)
        db.session.commit()
        response = jsonify({
            "historyId": new_history.id,  # Return the historyId of the newly created history
            "matches": [{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': float(similarity)
            } for proj, similarity in top_matches],
            "message": "Similarity Checked"
        })
        request_end = time.time()
        print(f"/index route total time (create): {request_end - request_start:.3f}s")
        rebuild_faiss_index_async()  # Rebuild index in background
        return response

@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    name = data.get('name')
    message = data.get('message')

    # Validate input
    if not name or not message:
        return jsonify({"error": "Name and message are required."}), 400

    # Create a new contact entry
    new_contact = Contact(
        name=name,
        message=message,
    )

    try:
        db.session.add(new_contact)
        db.session.commit()  # Save to database
        return jsonify({"message": "Messaged Sent!"}), 200
    except Exception as e:
        db.session.rollback()  # Rollback in case of error
        return jsonify({"error": str(e)}), 500  # Internal server error response


@app.route('/history', methods=['GET'])
@token_required
def history(current_user):
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    history_data = [{
        "id": h.id,
        "title": h.title,
        "summary": h.summary,
        "domain": h.domain,
        "timestamp": h.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "results": h.results  # Include the similarity results here
    } for h in user_history]
    return jsonify({"history": history_data}), 200

@app.route('/history/<int:history_id>', methods=['GET', 'DELETE', 'PUT'])
@token_required
def manage_specific_history(current_user, history_id):
    history_entry = History.query.filter_by(id=history_id, user_id=current_user.id).first()
    if not history_entry:
        return jsonify({"error": "History not found"}), 404

    if request.method == 'GET':
        return jsonify({
            "id": history_entry.id,
            "title": history_entry.title,
            "summary": history_entry.summary,
            "domain": history_entry.domain,
            "results": history_entry.results,
            "timestamp": history_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    elif request.method == 'DELETE':
        db.session.delete(history_entry)
        db.session.commit()
        return jsonify({"message": "Deleted"})
    elif request.method == 'PUT':
        data = request.get_json()
        history_entry.title = data.get('title', history_entry.title)
        history_entry.summary = data.get('summary', history_entry.summary)
        history_entry.domain = data.get('domain', history_entry.domain)
        db.session.commit()
        return jsonify({"message": "Updated successfully"})

@app.route('/api/similarity', methods=['POST'])
@token_required
def api_similarity(current_user):
    data = request.get_json()
    title = data.get('title', '').strip()
    summary = data.get('summary', '').strip()
    domain = data.get('domain', '').strip()
    top_matches = calculate_similarity(title, summary)
    results = [{
        'id': proj.id,
        'title': proj.title,
        'summary': proj.summary,
        'domain': proj.domain,
        'similarity': float(similarity)
    } for proj, similarity in top_matches]
    return jsonify({"matches": results})

@app.route('/feedback', methods=['POST'])
@token_required
def feedback(current_user):
    data = request.get_json()
    feedback_message = data.get('message', '').strip()  # Feedback message
    history_id = data.get('historyId', None)  # History ID to associate feedback with

    # Check if both feedback message and historyId are provided
    if not feedback_message or not history_id:
        return jsonify({"error": "Both message and historyId are required"}), 400

    # Find the corresponding history entry using historyId and userId
    history = History.query.filter_by(id=history_id, user_id=current_user.id).first()

    if not history:
        return jsonify({"error": "History not found"}), 404  # If history entry is not found

    # Save the feedback message into the database (assuming a Feedback model exists)
    new_feedback = Feedback(
        user_id=current_user.id,
        history_id=history.id,  # Associate feedback with the project
        message=feedback_message,
    )
    db.session.add(new_feedback)
    db.session.commit()

    # Return a success response
    return jsonify({
        "message": "Thankyou for your Feedback!"
    }), 200


@app.route('/help')
def help():
    return jsonify({"message": "Available: /login, /logout, /signup, /index, /history, /user"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='localhost', port=5000, debug=True)
