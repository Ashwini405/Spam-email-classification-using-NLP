# Peprocess the data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize words
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english') and word.isalnum()]
    return ' '.join(words)

data['processed_message'] = data['message'].apply(preprocess_text)

# Vectorize the text using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['processed_message'])

# Encode labels (spam=1, ham=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})
y = data['label']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test the model with a custom email
def predict_email(email):
    processed_email = preprocess_text(email)
    email_vector = vectorizer.transform([processed_email])
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
email = "Congratulations! You've won a free ticket to the Bahamas. Call now!"
print(f"Prediction: {predict_email(email)}")
