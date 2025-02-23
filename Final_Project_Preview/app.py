from flask import Flask, render_template, request, jsonify
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Directories for uploads and static files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to format chatbot response for better readability
def format_response(response):
    """
    Formats chatbot response for better readability.
    """
    return (
        response
        .replace("**", "")
        .replace("* ", "- ")
        .replace("1.", "\n1.")
        .replace("2.", "\n2.")
    )

# Home route
@app.route('/')
def home():
    """
    Serve the homepage (index.html).
    """
    return render_template('index.html')

# Chatbot route (renders the chatbot interface)
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """
    Serve the chatbot HTML page and handle user input.
    """
    if request.method == 'POST':
        user_message = request.form.get('user_message')
        if not user_message:
            return render_template('chatbot.html', response="Please enter a valid question.")
        
        # Vehicle-related instruction for the model
        instruction = (
            "You are an expert in vehicles and batteries. Answer questions related to vehicle maintenance, "
            "battery health, and performance improvements. Provide clear and structured responses using bullet points, "
            "headings, or numbered lists. Respond with 'I'm here to help with vehicle-related topics only.' for unrelated questions."
        )

        # Combine instruction with user input
        full_message = f"{instruction}\nUser: {user_message}\nAssistant:"

        try:
            # Use subprocess to call the gemma2:2b model via Ollama CLI
            process = subprocess.run(
                ['ollama', 'run', 'gemma2:2b'],
                input=full_message,
                text=True,
                capture_output=True,
                encoding='utf-8'  # Specify encoding as utf-8
            )

            response = process.stdout.strip()  # Extract the response from CLI output

        except Exception as e:
            return render_template('chatbot.html', response=f"Error interacting with gemma2: {str(e)}")

        # Format the response for better readability
        formatted_response = format_response(response)

        return render_template('chatbot.html', response=formatted_response)

    return render_template('chatbot.html')


# File upload route for battery analysis
@app.route('/upload', methods=['GET', 'POST'])
def upload_data():
    """
    Upload CSV or Excel file for battery data analysis.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Read uploaded file
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                data = pd.read_excel(filepath)
            else:
                return jsonify({"error": "Unsupported file format"}), 400

            # Perform analysis and predictions
            result, plot_path = analyze_battery_data(data)

            return render_template(
                'analysis.html',
                result=result,
                plot_path=plot_path
            )

        except Exception as e:
            return jsonify({"error": f"Failed to process file: {e}"}), 500

    return render_template('upload.html')


# Function to analyze battery data (for file upload)
def analyze_battery_data(data):
    """
    Analyze battery data and perform predictions.
    """
    # Example: Expect columns 'Cycles' and 'Capacity'
    if 'Cycles' not in data.columns or 'Capacity' not in data.columns:
        raise ValueError("Uploaded file must have 'Cycles' and 'Capacity' columns.")

    cycles = data['Cycles']
    capacity = data['Capacity']

    # Non-linear predictions (e.g., using a regression model)
    X = np.array(cycles).reshape(-1, 1)
    y = np.array(capacity)

    # Fit a simple regression model
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(cycles, capacity, color='blue', label='Observed Data')
    plt.plot(cycles, predictions, color='red', label='Predicted Trend')
    plt.title('Battery Capacity Analysis')
    plt.xlabel('Charge-Discharge Cycles')
    plt.ylabel('Capacity Retention (%)')
    plt.legend()
    plt.grid()

    # Save plot
    plot_path = os.path.join(app.static_folder, 'analysis.png')
    plt.savefig(plot_path)
    plt.close()

    return {"message": "Analysis complete"}, plot_path


# Main entry point for deployment on Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
