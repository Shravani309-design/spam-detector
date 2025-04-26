# Step 1: Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load and prepare data
file_name = 'email (1).csv'  # <-- Your local CSV file name
df = pd.read_csv(file_name, usecols=['Category', 'Message'])

# Clean data
df = df.dropna(subset=['Category', 'Message'])
df = df.rename(columns={'Category': 'label', 'Message': 'message'})
df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label'])

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Step 4: Create text features
vectorizer = CountVectorizer(stop_words='english', max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_vec)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Step 7: Prediction functions
def predict_spam(text):
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)[0]
    probability = model.predict_proba(processed_text)[0][1]
    return {
        'prediction': 'SPAM 🚨' if prediction == 1 else 'HAM ✅',
        'confidence': f"{probability:.0%}",
        'input_text': text[:100] + '...' if len(text) > 100 else text
    }

def classify_emails(emails):
    results = []
    for email in emails:
        result = predict_spam(email)
        results.append(result)
    return results

# Step 8: Example usage
test_emails = [
    "Congratulations! You've won $1,000,000! Click here to claim your prize!",
    "Hi John, just wanted to confirm our meeting tomorrow at 2 PM.",
    "Your package has been shipped. Tracking number: XYZ123456",
    "URGENT: Your bank account needs verification. Click immediately!"
]

print("\nAI-Powered Email Classification Results:")
classification_results = classify_emails(test_emails)
for result in classification_results:
    print(f"\n{result['prediction']} ({result['confidence']} confidence)")
    print(f"Text: {result['input_text']}")

# Additional function
def predict_email(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"

# Test examples
print(predict_email("You’ve won a $1000 gift card! Click now to claim!"))
print(predict_email("Meeting rescheduled to 3 PM. Please confirm."))
