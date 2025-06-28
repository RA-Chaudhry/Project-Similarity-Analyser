# 📘 Project Similarity Analyzer

**Project Similarity Analyzer** is an AI-powered tool developed as a Final Year Project to help identify the similarity between project proposals. It uses advanced NLP models like **SBERT** and **T5** to compare the semantic meaning of user-submitted projects against a database of existing projects, providing a similarity score. The system also supports **user authentication** through a modern **React JS frontend**, enabling users to manage their history and interactions securely.

---

## 🚀 Features

- 🔐 **User Authentication** (Login/Signup) via React frontend
- 🔍 **Semantic Similarity Detection** using **SBERT**
- 📄 **Automatic Summary Generation** using **T5 Transformer**
- 📊 Displays a **similarity score (0–100%)**
- 🧠 Compares user project with the most similar stored project
- 💾 Stores user history and feedback securely in the database
- ⚡ Fast and scalable vector search using **FAISS**

---

## 🛠️ Tech Stack

| Area       | Technology Used                          |
|------------|-------------------------------------------|
| Frontend   | React JS, HTML, CSS, JavaScript           |
| Backend    | Python, Flask                             |
| NLP Models | SBERT (Similarity), T5 (Summarization)    |
| Search     | FAISS (Vector similarity search)          |
| Auth       | React Forms + API integration             |
| Database   | SQLite / MySQL                            |
| Dev Tools  | Visual Studio Code, Git                   |

---

## 📂 Project Structure

```

ProjectSimilarityAnalyzer/
├── frontend/              # React JS frontend with login/signup
├── backend/               # Flask API and model logic
├── model/                 # SBERT & T5 model loaders
├── static/                # Frontend static assets (if any)
├── templates/             # Optional HTML templates
├── database/              # SQLite/MySQL schema and config
└── README.md              # Project documentation

```

---

## 📈 How It Works

1. Users **register or log in** through the React frontend.
2. After authentication, they enter a **project title and summary**.
3. Backend uses **SBERT** to extract embeddings from the text.
4. **FAISS** finds the most semantically similar project in the database.
5. The **T5 model** generates summaries for both user input and matched project.
6. The system calculates a **similarity score** and displays the results.
7. User activity is stored and can be accessed later under their profile.

---

## 🎯 Use Cases

- 💡 Students checking the uniqueness of project ideas
- 👨‍🏫 Teachers or evaluators reviewing submissions
- 🔬 Researchers verifying novelty
- 🛡️ Academic integrity and plagiarism prevention

---

## 📌 Future Improvements

- 🗂️ Upload and process full documents (PDF/DOC)
- ☁️ Integrate cloud storage (Firebase, AWS)
- 📊 Admin dashboard for analytics and reports
- 🔔 Notification system for alerts or duplicate matches

---

## 🧑‍💻 Author

**Muhammad Ramzan**  
🎓 BSCS (2021–2025)  
📧 rramzanch99@gmail.com  
🏫 Government College University, Faisalabad – Sahiwal Campus

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute with prope

