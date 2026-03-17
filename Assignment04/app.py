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
    app.run(host='0.0.0.0', port=5000)