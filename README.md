## Sentiment Model (Day 3)

Added a sentiment analysis training script to the project.

- **Script:** `train_sentiment.py`
- **Approach:** Trains a Logistic Regression classifier on a demo dataset using TF-IDF features.
- **Artifacts:** Saves the trained model and vectorizer into the `models/` directory (this folder is ignored in Git).
- **Helper:** Provides `predict_sentiment_demo(text)` for quick prediction tests.

### How to run
```bash
python3 train_sentiment.py
