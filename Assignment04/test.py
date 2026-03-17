import pytest
import joblib
import subprocess
import time
import requests
from score import score

@pytest.fixture
def model():
    """Fixture to load the saved model for testing."""
    return joblib.load('best_model.joblib')

# --- UNIT TESTS ---
def test_score(model):
    # Standard test strings
    spam_text = "URGENT! You have won a 1 week FREE membership in our prize jackpot!"
    ham_text = "Hey, are we still meeting for lunch tomorrow?"
    
    # 1. Smoke test: does the function produce some output without crashing
    pred, prop = score(spam_text, model, 0.5)
    
    # 2. Format test: are the input/output formats/types as expected
    assert isinstance(pred, bool)
    assert isinstance(prop, float)
    
    # 3. Sanity check: is prediction value 0 or 1 (boolean True/False)
    assert pred in [True, False]
    
    # 4. Sanity check: is propensity score between 0 and 1
    assert 0.0 <= prop <= 1.0
    
    # 5. Edge case input: if threshold is 0, prediction always becomes 1 (True)
    pred_edge_0, _ = score(ham_text, model, 0.0)
    assert pred_edge_0 is True
    
    # 6. Edge case input: if threshold is 1, prediction always becomes 0 (False)
    # (Assuming propensity is not exactly 1.0)
    pred_edge_1, prop_val = score(spam_text, model, 1.0)
    if prop_val < 1.0:
        assert pred_edge_1 is False
        
    # 7. Typical input: on obvious spam, prediction is 1 (True)
    pred_spam, _ = score(spam_text, model, 0.5)
    assert pred_spam is True
    
    # 8. Typical input: on obvious non-spam, prediction is 0 (False)
    pred_ham, _ = score(ham_text, model, 0.5)
    assert pred_ham is False

# --- INTEGRATION TEST (FLASK) ---
def test_flask():
    """Launches the app, tests the endpoint, and closes the app."""
    # Launch flask app
    process = subprocess.Popen(["python", "app.py"])
    
    # Give the server 2 seconds to start up
    time.sleep(2) 
    
    try:
        # Send POST request to the localhost endpoint
        response = requests.post(
            "http://127.0.0.1:5000/score",
            json={"text": "URGENT! You have won a prize!"}
        )
        
        # Test the response
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        
    finally:
        # Close the flask app using command line termination
        process.terminate()
        process.wait()

# --- INTEGRATION TEST (DOCKER) ---
def test_docker():
    """Launches the docker container, tests the endpoint, and closes the container."""
    container_name = "test-spam-container"
    
    # 1. Build the docker image
    subprocess.run(["docker", "build", "-t", "spam-classifier-app", "."], check=True)
    
    # Force remove the container if it already exists from a previous failed run
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    
    # Launch the docker container in the background
    subprocess.Popen([
        "docker", "run", "-p", "5000:5000", "--name", container_name, "spam-classifier-app"
    ])
    
    # Give the container and Flask app 5 seconds to fully boot up
    time.sleep(5) 
    
    try:
        # 2. Send a request to the localhost endpoint /score
        response = requests.post(
            "http://127.0.0.1:5000/score",
            json={"text": "URGENT! You have won a 1 week FREE membership in our prize jackpot!"}
        )
        
        # 3. Check if the response is as expected
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] is True  # Expected to be spam
        
    finally:
        # 4. Close and remove the docker container
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)