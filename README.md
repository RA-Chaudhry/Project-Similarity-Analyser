# 📘 Project Similarity Analyzer

**Project Similarity Analyzer** is a full-stack AI-powered web application designed as a Final Year Project to detect the similarity between project ideas. It uses NLP models to compare the meaning of submitted project summaries with existing ones in a database and calculates a similarity score. The system includes user **login/signup functionality** through a **React JS frontend**, and the backend is built with **Flask + SBERT + T5 + FAISS**.

---

## 🚀 Features

- 🔐 User Authentication (Login/Signup)
- 🔍 Semantic similarity detection with **SBERT**
- 📄 Auto summarization using **T5 Transformer**
- 📊 Shows similarity score (0–100%) with the closest matching project
- 🧠 Displays both input and matched project summaries
- 🗃️ Stores user history and results in a database
- ⚡ Fast comparison using **FAISS** vector search

---

## 🛠️ Tech Stack

| Area       | Technology Used                          |
|------------|-------------------------------------------|
| Frontend   | React JS (Vite), JavaScript, HTML, CSS    |
| Backend    | Python, Flask                             |
| NLP Models | SBERT (Similarity), T5 (Summarization)    |
| Search     | FAISS (Semantic vector search)            |
| Database   | SQLite (projects.db)                      |
| Auth       | Custom login/signup with API              |
| Tools      | VS Code, Git                              |

---

## 📂 Project Structure

```

ProjectSimilarityAnalyzer/
├── backend/
│   ├── app.py                  # Main Flask app
│   ├── ai\_projects.csv         # CSV of existing project data
│   ├── faiss.index             # Prebuilt FAISS index
│   ├── import\_project\_data.py  # Data import script
│   ├── project\_embeddings.npy  # SBERT embeddings
│   ├── project\_id\_mapping.json # ID to project mapping
│   ├── requirements.txt        # Python dependencies
│   ├── instance/
│   │   └── projects.db         # SQLite database
│   │   └── query.sql           # SQL schema and scripts
│   └── venv/                   # Python virtual environment
│
├── frontend/
│   ├── public/                 # Static public assets
│   ├── src/                    # React source files
│   ├── .env                    # Environment config
│   ├── index.html              # Main HTML file
│   ├── vite.config.js          # Vite configuration
│   ├── package.json            # NPM dependencies
│   └── README.md               # Frontend README (optional)
│
└── README.md                   # Main project documentation

````

---

## 📈 How It Works

1. User registers or logs in via the React frontend.
2. They submit a project title and summary.
3. Flask backend processes the input using **SBERT** embeddings.
4. **FAISS** performs semantic search against stored project vectors.
5. The **T5 model** summarizes both input and matched project.
6. A similarity score and comparison are displayed to the user.
7. The interaction is saved in the `projects.db` database.

---

## 📌 How to Run

### 🔧 Backend Setup
```bash
cd backend
python app.py
````

### 🌐 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Ensure backend (`localhost:5000`) and frontend (`localhost:5173`) are both running.

---

## 🎯 Use Cases

* Students checking if their project ideas are original
* Teachers evaluating submitted project proposals
* Researchers detecting topic overlaps
* Institutions reducing plagiarism

---

## 🧑‍💻 Author

**Muhammad Ramzan**
🎓 BSCS (2021–2025)
📧 [rramzanch99@gmail.com](mailto:rramzanch99@gmail.com)
🏫 Government College University, Faisalabad – Sahiwal Campus

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and distribute with proper credit.

```
