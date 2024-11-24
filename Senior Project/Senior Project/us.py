# Full-Stack Code

from flask import Flask, request, render_template_string, url_for
import webbrowser
import threading
from werkzeug.serving import make_server
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time


# Load traditional ML models
with open('naive_bayes_model.pkl', 'rb') as nb_file:
    naive_bayes_model = pickle.load(nb_file)

with open('decision_tree_model.pkl', 'rb') as dt_file:
    decision_tree_model = pickle.load(dt_file)

with open('random_forest_model.pkl', 'rb') as rf_file:
    random_forest_model = pickle.load(rf_file)

# Load TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Load tokenizer for LSTM
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    lstm_tokenizer = pickle.load(tokenizer_file)

lstm_model = load_model('lstm_model.keras')

# Recompile the model to reset the optimizer
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to detect phishing
def detect_phishing(email_content, algorithm):
    if algorithm == 'Naive Bayes':
        email_vector = tfidf_vectorizer.transform([email_content])
        prediction = naive_bayes_model.predict_proba(email_vector)[:, 0]
    elif algorithm == 'Decision Tree':
        email_vector = tfidf_vectorizer.transform([email_content])
        prediction = decision_tree_model.predict_proba(email_vector)[:, 0]
    elif algorithm == 'Random Forest':
        email_vector = tfidf_vectorizer.transform([email_content])
        prediction = random_forest_model.predict_proba(email_vector)[:, 0]
    
    elif algorithm == 'LSTM':
        # Convert email content to sequences using the tokenizer
        email_seq = lstm_tokenizer.texts_to_sequences([email_content])
        
        # Pad the sequences to the correct length
        email_seq_padded = pad_sequences(email_seq, maxlen=150)  # Adjust maxlen to match model training
        
        # Ensure the input has the correct shape (batch_size, sequence_length)
        email_seq_padded = np.array(email_seq_padded).reshape(1, -1)
        
        # Get the prediction from the LSTM model
        prediction = lstm_model.predict(email_seq_padded)[0]
    
    # Return the prediction value
    return prediction[0]


# Flask app setup
app = Flask(__name__)


# HTML template for "Detection Overview" page
detection_overview = '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Email Detection </title>
    <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to bottom right, #667eea, #764ba2);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            color: #333;
        }

        .container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 15px 25px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            width: 100%;
            animation: fadeIn 0.7s ease-in-out;
        }

        header {
            font-size: 2rem;
            font-weight: 700;
            color: #343a40;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }

        header::after {
            content: '';
            display: block;
            width: 60px;
            height: 4px;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 15px auto 0;
            border-radius: 3px;
        }

        .icon {
            font-size: 4rem;
            color: #764ba2;
            margin-bottom: 15px;
            animation: bounce 1.5s infinite;
        }

        .tabs {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            text-align: center;
        }

        .tab-item {
            flex: 1;
            text-align: center;
        }

        .tabs img {
            width: 70px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .tabs img:hover {
            transform: scale(1.1);
        }

        .tab-label {
            text-align: center;
            font-weight: bold;
            margin-top: 5px;
            color: #495057;
        }

        .info-content {
            font-size: 1.1rem;
            text-align: center;
            margin-top: 20px;
        }

        .info-content p {
            display: none;
        }

        .info-content p.active {
            display: block;
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.2s ease;
            margin-top: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        button:active {
            transform: translateY(0);
            box-shadow: none;
        }

        .about-us-content {
            display: none;
        }

        .about-us h3 {
            cursor: pointer;
            color: #764ba2;
            text-align: center;
            margin-top: 30px;
        }

        .team-members {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }

        .team-member {
            text-align: center;
        }

        .team-member h4 {
            font-size: 1.2rem;
            color: #764ba2;
        }

        footer {
            margin-top: 30px;
            font-size: 0.9rem;
            color: #6c757d;
            text-align: center;
        }

        /* Keyframes for Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }

            50% {
                transform: translateY(-10px);
            }
        }
    </style>
    <script>
        function showContent(id) {
            var contents = document.querySelectorAll('.info-content p');
            contents.forEach(function(content) {
                content.classList.remove('active');
            });
            document.getElementById(id).classList.add('active');
        }

        function toggleAboutUs() {
            var aboutContent = document.getElementById('about-us-content');
            if (aboutContent.style.display === 'none' || aboutContent.style.display === '') {
                aboutContent.style.display = 'block';  // Show the content when clicked
            } else {
                aboutContent.style.display = 'none';  // Hide the content when clicked again
            }
        }
    </script>
</head>

<body>
    <div class="container">
        <header>
            <i class="fas fa-shield-alt icon"></i>
            Phishing Defense Suite
        </header>
        <div class="tabs">
             <div class="tab-item">
                <img src="{{ url_for('static', filename='image2.webp') }}" alt="Probability Icon" onclick="showContent('probability')">
                <p class="tab-label">Probability Analysis</p>
            </div>
            <div class="tab-item">
                <img src="{{ url_for('static', filename='image3.webp') }}" alt="Real-Time Icon" onclick="showContent('realtime')">
                <p class="tab-label">Real-Time Detection</p>
            </div>
            <div class="tab-item">
                <img src="{{ url_for('static', filename='image4.webp') }}" alt="ML Icon" onclick="showContent('ml')">
                <p class="tab-label">Machine Learning</p>
            </div>
        </div>
        <div class="info-content">
            <p id="probability">Our probability analysis enhances detection by estimating the risk level of each email, supporting the chosen algorithm with an added layer of predictive accuracy.</p>
            <p id="realtime">Our real-time phishing detection system analyzes emails as they are received, identifying threats instantly to keep users safe.</p>
            <p id="ml">Our machine learning technology identifies phishing by recognizing subtle patterns in email content, continuously refining its accuracy as it learns from new data.</p>
        </div>
        <a href="/form"><button><i class="fas fa-arrow-right"></i> Proceed to Check for Phishing</button></a>

        <!-- Clickable About Us Section with Team Members -->
        <div class="about-us">
            <h3 onclick="toggleAboutUs()">About Us</h3>
            <div id="about-us-content" class="about-us-content" style="display:none;">
                <p>We are a passionate team focused on delivering cutting-edge phishing detection technologies to safeguard users from cyber threats and ensure a secure online experience.</p>
                <h4 style="text-align:center; color:#764ba2;">Team Members</h4>
                <div class="team-members">
                    <div class="team-member">
                        <h4>Latia Maree</h4>
                        <p>IT</p>
                        <p>Front-End Developer</p>
                    </div>
                    <div class="team-member">
                        <h4>Kensley Benjamin</h4>
                        <p>Cybersecurity</p>
                        <p>Back-End Developer</p>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Phishing Detection &copy; 2024 | Stay Safe Online
        </footer>
    </div>

</body>

</html>
'''


# HTML template for the phishing form page
form_page = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Email Detection</title>
    <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to bottom right, #667eea, #764ba2);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            color: #333;
        }

        .container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 15px 25px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            width: 100%;
            animation: fadeIn 0.7s ease-in-out;
        }

        header {
            font-size: 2rem;
            font-weight: 700;
            color: #343a40;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }

        header::after {
            content: '';
            display: block;
            width: 60px;
            height: 4px;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 15px auto 0;
            border-radius: 3px;
        }

        .icon {
            font-size: 4rem;
            color: #764ba2;
            margin-bottom: 15px;
            animation: bounce 1.5s infinite;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            font-weight: 500;
            margin-bottom: 8px;
            color: #495057;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            font-size: 1.1rem;
            transition: border-color 0.3s ease;
            resize: none;
            min-height: 180px;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
        }

        textarea:focus {
            border-color: #764ba2;
            outline: none;
        }

        .toggle-button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.2s ease;
            margin-top: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        .toggle-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        .toggle-button:active {
            transform: translateY(0);
            box-shadow: none;
        }

        #algorithm-options {
            display: none;
            margin-top: 20px;
        }

        .custom-radio-wrapper {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            color: white;
        }

        .custom-radio-wrapper input {
            margin-right: 10px;
        }

        #naive_bayes + label,
        #decision_tree + label,
        #random_forest + label,
        #lstm + label {
            background: linear-gradient(to right, #667eea, #764ba2);
            padding: 10px;
            border-radius: 5px;
            width: 100%;
            color: white;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.2s ease;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        #naive_bayes + label:hover,
        #decision_tree + label:hover,
        #random_forest + label:hover,
        #lstm + label:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.2s ease;
            margin-top: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        button:active {
            transform: translateY(0);
            box-shadow: none;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 5px solid #764ba2;
            border-radius: 10px;
            font-size: 1.1rem;
            text-align: left;
            animation: fadeIn 0.7s ease-in-out;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .result.negative {
            background-color: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }

        .result h3 {
            margin: 0;
            font-weight: 600;
        }

        footer {
            margin-top: 30px;
            font-size: 0.9rem;
            color: #6c757d;
            text-align: center;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            header {
                font-size: 1.8rem;
            }

            textarea {
                min-height: 140px;
            }

            button {
                font-size: 1rem;
            }
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }

            50% {
                transform: translateY(-10px);
            }
        }
    </style>

    <script>
        console.log("JavaScript loaded!");  // Debugging message

        function toggleAlgorithmOptions() {
            var options = document.getElementById('algorithm-options');
            var checkButton = document.getElementById('check-phishing-btn');
            if (options.style.display === 'none' || options.style.display === '') {
                options.style.display = 'block';
                checkButton.disabled = false;  // Enable the "Check for Phishing" button after selecting algorithm
            } else {
                options.style.display = 'none';
                checkButton.disabled = true;  // Disable button if options are hidden
            }
        }

        function validateAlgorithmSelection(event) {
            console.log("Validating algorithm selection...");  // Debugging step
            const algorithms = document.getElementsByName('algorithm');
            let algorithmSelected = false;

            for (const algorithm of algorithms) {
                if (algorithm.checked) {
                    algorithmSelected = true;
                    break;
                }
            }

            if (!algorithmSelected) {
                event.preventDefault();  // Prevent form submission
                console.log("No algorithm selected.");  // Debugging log
                alert("Please select an algorithm before submitting.");  // Alert the user
                return false;  // Block submission
            }
            console.log("Algorithm selected.");  // Debugging log
            return true;  // Allow form submission
        }
    </script>
</head>

<body>

    <div class="container">
        <header>
            <i class="fas fa-shield-alt icon"></i>
            Phishing Email Detection
        </header>
        <!-- Ensure the form uses the correct validation function -->
        <form action="/result" method="POST" onsubmit="validateAlgorithmSelection(event)">
            <div class="form-group">
                <label for="email">Email Content:</label>
                <textarea id="email" name="email" placeholder="Paste the email content here..." required></textarea>
            </div>

            <!-- Toggle Button for Algorithm Selection -->
            <button id="choose-algorithm-btn" type="button" class="toggle-button" onclick="toggleAlgorithmOptions()">
                <i class="fas fa-cogs"></i> Choose Algorithm
            </button>

            <!-- Hidden Algorithm Options -->
            <div id="algorithm-options" style="display: none;">
                <div class="custom-radio-wrapper">
                    <input type="radio" id="naive_bayes" name="algorithm" value="Naive Bayes" checked>
                    <label for="naive_bayes">Naive Bayes</label>
                </div>
                <div class="custom-radio-wrapper">
                    <input type="radio" id="decision_tree" name="algorithm" value="Decision Tree">
                    <label for="decision_tree">Decision Tree</label>
                </div>
                <div class="custom-radio-wrapper">
                    <input type="radio" id="random_forest" name="algorithm" value="Random Forest">
                    <label for="random_forest">Random Forest</label>
                </div>
                <div class="custom-radio-wrapper">
                    <input type="radio" id="lstm" name="algorithm" value="LSTM">
                    <label for="lstm">LSTM Neural Network</label>
                </div>
            </div>

            <!-- Initially disabled "Check for Phishing" button -->
            <button id="check-phishing-btn" type="submit" disabled>
                <i class="fas fa-search"></i> Check for Phishing
            </button>
        </form>

        {% if result %}
        <div class="result {{ 'negative' if 'No' in result else '' }}">
            <h3><i class="fas {{ 'fa-times-circle' if 'No' in result else 'fa-check-circle' }}"></i> {{ result }}</h3>
        </div>
        {% endif %}

        <footer>
            Phishing Detection &copy; 2024 | Stay Safe Online
        </footer>
    </div>

</body>

</html>
'''

# HTML template for the result page
result_page = '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Detection</title>
    <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to bottom right, #667eea, #764ba2);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            color: #333;
        }

        .container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 15px 25px rgba(0, 0, 0, 0.2);
            max-width: 600px;
            width: 100%;
            animation: fadeIn 0.7s ease-in-out;
        }

        header {
            font-size: 2rem;
            font-weight: 700;
            color: #343a40;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
        }

        header::after {
            content: '';
            display: block;
            width: 60px;
            height: 4px;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 15px auto 0;
            border-radius: 3px;
        }

        .icon {
            font-size: 4rem;
            color: #764ba2;
            margin-bottom: 15px;
            animation: bounce 1.5s infinite;
        }

        .result {
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 30px;
            padding: 15px;
            border-radius: 8px;
            color: white;
        }

        .result.safe {
            background-color: #28a745; /* Green for safe */
        }

        .result.phishing {
            background-color: #dc3545; /* Red for phishing */
        }

        .prediction {
            font-size: 1.1rem;
            margin-top: 10px;
            color: #333;
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.2rem;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.4s ease, transform 0.2s ease;
            margin-top: 20px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        button:active {
            transform: translateY(0);
            box-shadow: none;
        }

        footer {
            margin-top: 30px;
            font-size: 0.9rem;
            color: #6c757d;
            text-align: center;
        }

        /* Keyframes for Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes bounce {
            0%, 100% {
                transform: translateY(0);
            }
            50% {
                transform: translateY(-10px);
            }
        }

    </style>
</head>

<body>

    <div class="container">
        <header>
            <i class="fas fa-shield-alt icon"></i>
            Phishing Detection Result
        </header>
        <div class="result {{ 'safe' if prediction_percentage < 50 else 'phishing' }}">
            <h3>{{ result }}</h3>
            <p class="prediction">Prediction Probability: {{ prediction_percentage }}%</p>
        </div>
        <div class="text-center">
            <a href="/form"><button><i class="fas fa-arrow-right"></i> Check Another Email</button></a>
            <a href="/"><button><i class="fas fa-info-circle"></i> Homepage</button></a>
        </div>

        <footer>
            Phishing Detection &copy; 2024 | Stay Safe Online
        </footer>
    </div>

</body>
</html>
'''


# Flask routes
@app.route('/')
def detection_overview_page():
    return render_template_string(detection_overview)

@app.route('/form')
def phishing_form():
    return render_template_string(form_page)

@app.route('/result', methods=['POST'])
def phishing_result():
    email_content = request.form['email']
    algorithm = request.form['algorithm']

    # Validate email content
    if not email_content.strip():
        return render_template_string(result_page, 
                                      result='Invalid email content provided.', 
                                      prediction_percentage='N/A')

    if not algorithm:
        return render_template_string(result_page, 
                                      result='No algorithm selected.', 
                                      prediction_percentage='N/A')

    try:
        # Debug: Print the email content and selected algorithm
        print(f"Email content: {email_content}")  
        print(f"Algorithm selected: {algorithm}")  

        # Get the prediction from the chosen algorithm
        prediction = detect_phishing(email_content, algorithm)
        
        # Debug: Print raw prediction value for checking
        print(f"Prediction (raw output): {prediction}")

    except Exception as e:
        return render_template_string(result_page, 
                                      result=f'Error during prediction: {str(e)}', 
                                      prediction_percentage='N/A')

    # Convert prediction to percentage for display
    prediction_percentage = round(float(prediction) * 100, 2)

    # Debugging: Show output range
    print(f"Prediction percentage for content: {prediction_percentage}%")

    # Determine if it’s phishing or safe based on the prediction threshold
    # Lowering the threshold to 0.5 for phishing classification
    if prediction >= 0.5:  # Adjusted threshold
        result = 'No, it is a phishing email.'
    else:
        result = 'Yes, it is a safe email.'

    # Debug: print final result and prediction percentage
    print(f"Result: {result}, Prediction percentage: {prediction_percentage}")

    return render_template_string(result_page, 
                                  result=result, 
                                  prediction_percentage=prediction_percentage)


# Start Flask in the background
class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.srv = make_server('127.0.0.1', 5000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()


# Start the server
server = ServerThread(app)
server.start()

# Open the page in a browser
open_browser()