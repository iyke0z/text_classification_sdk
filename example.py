from text_classifier_sdk.classifier import TextClassifier

# Define labels
labels = ["positive", "negative", "neutral"]

# Initialize classifier
classifier = TextClassifier(labels)

# Train with user-supplied dataset
classifier.train("train.csv", "test.csv")

# Make a prediction
print(classifier.predict("This product is amazing!"))  # Example output: "positive"
