%pip install pandas
%pip install numpy
%pip install keras
%pip install --upgrade tensorflow
%pip install wordcloud
%pip install flask
%pip install seaborn
%pip install scikit-learn==1.4.2



from flask import Flask, render_template_string
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, LSTM, Bidirectional, SimpleRNN
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from wordcloud import WordCloud
import threading
import webbrowser
from werkzeug.serving import make_server
from sklearn.metrics import classification_report, confusion_matrix


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("Phishing_Email.csv")

nRow, nCol = df.shape
print(f'Rows: {nRow} and Cols: {nCol}')

df.head(10)


df.isnull().sum()


#drop null vals
df.isnull().sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("New data length:", len(df), "\n")


df.isnull().sum()


print("Dimension of row:",df.shape)


email_type = df['Email Type'].value_counts()
plt.bar(email_type.index, email_type.values, color=['darkblue', 'darkred'])
plt.xlabel('Email Type')
plt.ylabel('Number')
plt.title('Split of Email Types')

# Display the plot
plt.show()


plt.pie(email_type, labels=email_type.index, colors=['darkblue', 'darkred'], autopct='%1.1f%%')

# Display the pie chart
plt.show()


wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
safe_type = df[df['Email Type'] == 'Safe Email']['Email Text'].astype(str).values
phishing_type = df[df['Email Type'] == 'Phishing Email']['Email Text'].astype(str).values

safe_type = ''.join(safe_type)
wordcloud.generate(safe_type)

plt.figure(figsize=(6, 7))
plt.imshow(wordcloud)
plt.title('Safe Visualization')
plt.axis("off")
plt.show()


phishing_type = ''.join(phishing_type)
wordcloud.generate(phishing_type)

plt.figure(figsize=(6, 7))
plt.imshow(wordcloud)
plt.title('Phishing Visualization')
plt.axis("off")
plt.show()


# Vectorize email content using TF-IDF
tf = TfidfVectorizer(stop_words="english",max_features=10000)
feature_x = tf.fit_transform(df["Email Text"]).toarray()

y_tf = np.array(df['Email Type'])

# Split the dataset into training and test sets
X_train,X_test,y_train,y_test = train_test_split(feature_x,y_tf,train_size=0.8,random_state=0)


y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

print("Instances per label in training:", y_train_series.value_counts())
print("Instances per label in test:", y_test_series.value_counts())


print(X_train[:2])
print(X_train.shape)


print(y_train)
print(y_train.shape)


# Naive Bayes
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Define and fit the TF-IDF vectorizer using the 'Email Text' column
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_vect = tfidf_vectorizer.fit_transform(df["Email Text"])  # Transform the full text column for training and testing

# Convert the labels into numpy array
y = np.array(df['Email Type'])

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train_vect, X_test_vect, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=0)

# Train the Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_vect, y_train)

# Predict on the test set using Naive Bayes classifier
pred_nav = nb.predict(X_test_vect)

# Print accuracy
print(f"Accuracy from Naive Bayes: {accuracy_score(y_test, pred_nav) * 100:.2f} %")

# Print F1 score (weighted average for multi-class classification)
print(f"F1 score from Naive Bayes: {f1_score(y_test, pred_nav, average='weighted') * 100:.2f} %")

# Print the classification report
print("Classification Report:\n", classification_report(y_test, pred_nav))

# Save the Naive Bayes model
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(nb, model_file)

# Save the TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Additional evaluation metrics for clarity
print("Naive Bayes Evaluation Metrics:")
print(classification_report(y_test, pred_nav, target_names=['Safe', 'Phishing']))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_nav))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
clf_nav = confusion_matrix(y_test, pred_nav)

# Display confusion matrix
cx_ = ConfusionMatrixDisplay(confusion_matrix=clf_nav, display_labels=['phishing_mail', 'safe_mail'])
cx_.plot(cmap='viridis') 
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Decision Tree
dtr = DecisionTreeClassifier()
dtr = DecisionTreeClassifier() # Creating the classifier object
dtr.fit(X_train, y_train)

#Prediction on the test set
pred_dtr = dtr.predict(X_test)

#Performance evaluation
print(f"Accuracy from Decision Tree: {accuracy_score(y_test, pred_dtr) * 100:.2f} %")
print(f"F1 score from Decision Tree: {f1_score(y_test, pred_dtr, average='weighted') * 100:.2f} %")
print("Classification report:\n", classification_report(y_test, pred_dtr))


print("Decision Tree Evaluation Metrics:")
print(classification_report(y_test, pred_dtr, target_names=['Safe', 'Phishing']))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_dtr))
#Confusion Matrix
clf_dtr = confusion_matrix(y_test, pred_dtr)
ConfusionMatrixDisplay(clf_dtr, display_labels=['phishing_mail', 'safe_mail']).plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

import pickle

# Saving the Decision Tree model to 'decision_tree_model.pkl'
with open('decision_tree_model.pkl', 'wb') as dt_file:
    pickle.dump(dtr, dt_file)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Initialize and train the Random Forest model
rnf = RandomForestClassifier()
rnf.fit(X_train, y_train)

# Prediction
pred_rnf = rnf.predict(X_test)

# Performance metrics
print(f"Accuracy from Random Forest: {accuracy_score(y_test, pred_rnf) * 100:.2f}%")
print(f"F1 score from Random Forest: {f1_score(y_test, pred_rnf, average='weighted') * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, pred_rnf, target_names=['Safe', 'Phishing']))

# Confusion Matrix Display
clf_rnf = confusion_matrix(y_test, pred_rnf)
ConfusionMatrixDisplay(confusion_matrix=clf_rnf, display_labels=['phishing_mail', 'safe_mail']).plot(cmap='viridis')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Save the Random Forest model
with open('random_forest_model.pkl', 'wb') as rf_file:
    pickle.dump(rnf, rf_file)  # Save the trained model


# SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Step 1: Initialize individual models, including SGDClassifier with correct 'log_loss' parameter
nb_model = MultinomialNB()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)  # Corrected parameter to 'log_loss'

# Step 2: Create an ensemble model using VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('dt', dt_model),
    ('rf', rf_model),
    ('sgd', sgd_model)
], voting='soft')  # Soft voting for probability averaging

# Step 3: Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
ensemble_pred = ensemble_model.predict(X_test)

# Step 5: Evaluate the ensemble model
print(f"Ensemble Model Accuracy: {accuracy_score(y_test, ensemble_pred) * 100:.2f}%")
print(f"Ensemble Model F1 Score: {f1_score(y_test, ensemble_pred, average='weighted') * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, ensemble_pred))

# Display confusion matrix for detailed error analysis
conf_matrix = confusion_matrix(y_test, ensemble_pred)
print("Confusion Matrix:\n", conf_matrix)


models = ['Naive Bayes', 'Decision Tree', 'Random Forest', 'SGD Classifier']
accuracies = [95.12, 92.70, 96.54, 97.10]

plt.bar(models, accuracies, color=['darkblue', 'darkred', 'darkorange', 'darkgreen'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models')

# Display the plot
plt.show()


# LSTM/ RNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

max_length = 150

tk = Tokenizer()
tk.fit_on_texts(df['Email Text'])

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tk, tokenizer_file)

# Convert text to sequences and pad to max_length
sequences = tk.texts_to_sequences(df['Email Text'])
vector = pad_sequences(sequences, padding='post', maxlen=max_length)  # Ensure maxlen is consistent here


x = np.array(vector)
y = np.array(df["Email Type"])

X_train, X_test, y_train, y_test = train_test_split(vector,df['Email Type'], test_size=0.2, random_state =0)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import numpy as np

max_length = 150  # Define max length for padding sequences

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tk.word_index)+1, output_dim=100))  # Removed input_length
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build the model by calling it on sample data to initialize weights
sample_data = np.zeros((1, max_length))  # Create a sample batch with the correct input shape
model(sample_data)  # This initializes the model weights

# Now save the model
model.save('lstm_model.keras')
model.summary()