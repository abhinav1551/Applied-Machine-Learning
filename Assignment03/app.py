from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the model globally so it only loads once when the app starts
MODEL = joblib.load('best_model.joblib')

@app.route('/score', methods=['POST'])
def score_endpoint():
    # Extract JSON data from the request
    data = request.get_json()
    text = data.get('text', '')
    
    # Score the text (using 0.5 as a standard threshold)
    prediction, propensity = score(text, MODEL, 0.5)
    
    # Return the response as JSON
    return jsonify({
        'prediction': prediction,
        'propensity': propensity
    })

if __name__ == '__main__':
    # Run on localhost (127.0.0.1) port 5000
    app.run(host='127.0.0.1', port=5000)