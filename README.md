# 🧠 AI-Powered Productivity Assistant

A personal productivity assistant that helps users manage tasks, track motivation through journal entries, and get intelligent recommendations based on sentiment analysis.

---

## 🚀 Overview
This project integrates **task management**, **journal tracking**, and **AI-based sentiment analysis** into one lightweight productivity system.

- Users can add and track daily tasks.
- Journal entries are automatically analyzed for sentiment (positive, neutral, negative).
- The system uses this sentiment to provide coding and wellness recommendations.
- Built with **Python**, **SQLite**, and **scikit-learn**.

---

## 🧩 Key Features
- 🗂️ **Task Manager** – Add, view, and mark tasks as complete.
- 📝 **Journal Tracker** – Write short daily reflections.
- 💡 **Sentiment Analysis** – TF-IDF + Logistic Regression model (trained on IMDB dataset, 88% accuracy).
- 🎯 **3-way Sentiment Mapping** – Converts probabilities to *positive*, *neutral*, or *negative*.
- 🧘 **Rule-based Recommender** – Suggests breaks, coding challenges, or wellness activities based on your mood.
- 🧹 **Text Preprocessing** – Custom `clean_text()` removes noise, punctuation, and URLs.

---

## 🧠 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | **88.28%** |
| F1-score | **0.88** |
| Dataset | IMDB Movie Reviews (50K samples) |

---

## ⚙️ Tech Stack
**Languages & Tools:**  
`Python` · `SQLite` · `scikit-learn` · `pandas` · `datasets` · `TfidfVectorizer`

---

## 🔗 Project Structure
```
ai_productivity_assistant/
├── app/
│   └── task.py
├── models/         # (ignored in git)
├── .gitignore
├── README.md
├── requirements.txt
├── train_imdb.py
└── train_sentiment.py
```

---

## 🧪 How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/Medha-glitch/ai_productivity_assistant.git
   cd ai_productivity_assistant
2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
4. **Train or load model**
   ```bash
   python train_imdb.py --sample-frac 0.1
5. **Run the app**
   ```bash
   python -m app.task

---

## 🧭 Roadmap
- Build sentiment model (TF-IDF + Logistic Regression)
- Integrate with journal entries
- Add 3-way sentiment labeling
- Store sentiment probabilities
- Implement rule-based recommender
- Flask-based web dashboard
- Daily sentiment analytics
- Deploy to Render/Heroku

---

## 📚 Learnings
- Applied **NLP preprocessing**, **TF-IDF vectorization**, and model training.
- Integrated **ML predictions** into a functional backend.
- Designed modular, extensible architecture for later **Flask UI integration**.

---

## 👩‍💻 Author
**Medha Sharma** B.Tech. Computer Science and Engineering | Thapar Institute of Engineering and Technology  
[LinkedIn](https://www.linkedin.com/in/medha-sharma-b024b0252/) · [GitHub](https://github.com/Medha-glitch)
