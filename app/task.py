import sqlite3
from datetime import datetime
import re
from string import punctuation
import pickle
from config import MODEL_DIR, DB_FILE, POS_THRESH, NEG_THRESH


# -------------------------
# Model loading (load once)
# -------------------------
MODEL_VEC = None
MODEL_CLF = None

def load_model_once():
    global MODEL_VEC, MODEL_CLF
    try:
        with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f:
            MODEL_VEC = pickle.load(f)
        with open(f"{MODEL_DIR}/sentiment_model.pkl", "rb") as f:
            MODEL_CLF = pickle.load(f)
        print("Loaded sentiment model into memory.")
    except Exception as e:
        MODEL_VEC, MODEL_CLF = None, None
        print("Could not load sentiment model at startup:", e)

# call at import time
load_model_once()

def predict_sentiment_cached(text: str):
    """
    Returns (label, prob_positive) or (None, None) if model not loaded.
    """
    if MODEL_VEC is None or MODEL_CLF is None:
        return None, None
    t = clean_text(text)
    X = MODEL_VEC.transform([t])
    proba = MODEL_CLF.predict_proba(X)[0]  # [prob_neg, prob_pos]
    label = "positive" if proba[1] >= 0.5 else "negative"
    return label, float(proba[1])


# -------------------------
# DB helpers
# -------------------------
def connect_db():
    conn = sqlite3.connect("DB_FILE")
    return conn

# Creating tasks table
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        date_added TEXT
    )
""")
    conn.commit()
    conn.close()

# Journal table 
def create_journal_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry TEXT NOT NULL,
            date_added TEXT
        )
    """)
    conn.commit()
    conn.close()

def ensure_journal_sentiment_column():
    """
    Adds a 'sentiment' TEXT column to journal table if it doesn't exist.
    Safe to call multiple times.
    """
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT sentiment FROM journal LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist — add it
        cursor.execute("ALTER TABLE journal ADD COLUMN sentiment TEXT")
    conn.commit()
    conn.close()

def ensure_journal_sentiment_cols():
    conn = connect_db()
    cursor = conn.cursor()
    # Try selecting both columns - if it fails, add the missing one
    try:
        cursor.execute("SELECT sentiment, sentiment_prob FROM journal LIMIT 1")
    except sqlite3.OperationalError:
        # Add sentiment if missing
        try:
            cursor.execute("ALTER TABLE journal ADD COLUMN sentiment TEXT")
        except Exception:
            pass
        # Add sentiment_prob if missing
        try:
            cursor.execute("ALTER TABLE journal ADD COLUMN sentiment_prob REAL")
        except Exception:
            pass
    conn.commit()
    conn.close()


# -------------------------
# Tasks
# -------------------------
def add_task(task):
    task = task.strip()
    if not task:
        print("❗ Task cannot be empty.")
        return
    conn = connect_db()
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO tasks (task, date_added) VALUES (?, ?)", (task, date))
    conn.commit()
    conn.close()
    print("✅ Task added.")


def complete_task(task_id):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return False
    
    if row[0] == "done":
        conn.close()
        return False
    
    cursor.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return True

#Viewing all tasks
def view_tasks():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, task, status, date_added FROM tasks ORDER BY date_added DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


# -------------------------
# Journal
# -------------------------
def label_from_prob(p: float, pos_thresh=POS_THRESH, neg_thresh=NEG_THRESH):
    if p is None:
        return None
    if p >= pos_thresh:
        return "positive"
    if p <= neg_thresh:
        return "negative"
    return "neutral"

def add_entry(entry):
    entry = entry.strip()
    if not entry:
        print("Entry not empty!!!!")
        return
    
    # Predict sentiment (cached)
    raw_label, prob = predict_sentiment_cached(entry) # raw_label is 'positive'/'negative' or None
    ternary_label = label_from_prob(prob) if prob is not None else None

    conn = connect_db()
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute(
            "INSERT INTO journal (entry, date_added, sentiment, sentiment_prob) VALUES (?, ?, ?, ?)",
            (entry, date, ternary_label, prob),
        )
    except sqlite3.OperationalError:
        # Fallback in case column is missing
        cursor.execute("INSERT INTO journal (entry, added) VALUES (?, ?)", (entry, date))
    conn.commit()
    conn.close()

    if ternary_label is not None:
        print(f"Entry saved. Sentiment: {ternary_label} (positive_prob={prob:.2f})")
    else:
        print("Entry saved. (model not available)")

def view_entries():
    """
    Tries to include sentiment column if it exists; falls back otherwise.
    Returns rows.
    """
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, entry, date_added, sentiment FROM journal ORDER BY date_added DESC")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # sentiment column not present
        cursor.execute("SELECT id, entry, date_added FROM journal ORDER BY date_added DESC")
        rows = cursor.fetchall()
    conn.close()
    return rows

# ------------ Simple rule based Recommender --------
LEETCODE_SUGGESTIONS = {
    "easy": [
        "Two Sum (Array, HashMap)",
        "Reverse Linked List (Linked List)",
        "Valid Parentheses (Stack)"
    ],
    "medium": [
        "Number of Islands (DFS/BFS)",
        "Merge Intervals (Sorting + Greedy)",
        "Binary Tree Level Order Traversal (BFS)"
    ],
    "hard": [
        "LFU Cache (Design + Hash + DLL)",
        "Word Ladder II (BFS + Backtracking)",
        "Knapsack Problem (Dynamic Programming)"
    ]
}

def recommend_for_latest(user_pref=None):
    """
    Looks at the most recent journal entry and returns recommendations.
    user_pref can be 'coding' or 'wellness' to bias suggestions.
    """

    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT entry, sentiment, sentiment_prob FROM journal ORDER BY date_added DESC LIMIT 1")
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        row = None
    conn.close()

    if row in None:
        return {
            "message": "No journal entries yet.",
            "Suggestions": ["Write your first journal entry!!"]
        }
    entry, sentiment, prob = row
    suggestions = []
    message = f"Latest sentiment: {sentiment} (p_pos={prob:.2f})"

    if sentiment == "negative":
        suggestions += [
            "Take a 10-min walk or short break",
            "Try deep breathing journaling your thoughts"
        ] 
        if user_pref == "coding":
            suggestions += LEETCODE_SUGGESTIONS["easy"][:2]

    elif sentiment == "neutral":
        suggestions += [
            "Organize your workspace or plan your next session",
            "Do a short Pomodoro (25/5)"
        ]
        if user_pref == "coding":
            suggestions += LEETCODE_SUGGESTIONS["medium"][:2]

    elif sentiment == "Positive":
        suggestions += [
            "Ride the momentum - do one focused 50-min work block",
            "Reward yourself after completing one major task"
        ]
        if user_pref == "coding":
            suggestions += LEETCODE_SUGGESTIONS["hard"][:2]
    else:
        suggestions = ["Sentiment unknown - reflect for a moment or try writing again."]
    
    return{
        "entry": entry,
        "sentiment" : sentiment,
        "probability": prob,
        "suggestions": suggestions

    }


# ------ Text Cleaning --------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
WHITESPACE_RE = re.compile(r'\s+')

PUNCT_TO_REMOVE = ''.join(ch for ch in punctuation if ch != "'")
PUNCT_RE = re.compile('[' + re.escape(PUNCT_TO_REMOVE) + ']')

def clean_text(s: str) -> str:
    """
    Lowercase; remove URLs; remove punctuation except apostrophes;
    collapse whitespace; trim.
    """
    if not s:
        return ""
    s = s.lower()
    s = URL_RE.sub('', s)
    s = PUNCT_RE.sub('', s)
    s = WHITESPACE_RE.sub(' ', s).strip()
    return s

def preprocess_entries():
    """
    Fetch journal entries, return list of tuples:
    (id, raw_entry, cleaned_entry, date_added)
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, entry, date_added FROM journal ORDER BY date_added DESC")
    rows = cursor.fetchall()
    conn.close()

    processed = []
    for rid, entry, ts in rows:
        cleaned = clean_text(entry)
        processed.append((rid, entry, cleaned, ts))
    return processed


# -------------------------
# CLI / quick tests when running module
# -------------------------
if __name__ == "__main__":
    create_table()
    create_journal_table()
    ensure_journal_sentiment_column() # <<-- ensure column exists before inserts
    ensure_journal_sentiment_cols()

    # Tasks
    add_task("Prepare resume for placements")
    add_task("Solve 2 LeetCode problems")
    add_task("Apply for Deloitte")
    add_task("Revise SQL Joins")

    # Journal
    add_entry("Feeling a bit stressed but motivated to code.")
    add_entry("Had a productive day, solved 3 DSA questions!")

    print("\nTasks:")
    for t in view_tasks():
        print(t)

    print("\nJournal Entries:")
    for j in view_entries():
        print(j)

    # Quick cleaner checks
    print("\nCleaner test 1:", clean_text("So stressed!!!  Visit https://x.y/zz "))
    print("Cleaner test 2:", clean_text("I can't focus...   "))
    print("Cleaner test 3:", clean_text("Great day :) #winning"))
    print("\nProcessed journal entries:")
    for r in preprocess_entries():
        print(r)
