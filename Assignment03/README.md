# Assignment 03: Testing & Model Serving  
## SMS Spam Classifier Deployment  

---

## Objective  

The goal of this assignment was to take a trained machine learning model and convert it into a reliable, testable, and deployable REST API.  

Instead of stopping at Jupyter Notebook experimentation, this project demonstrates a complete ML engineering workflow including automated testing and Flask deployment.

---

## What I Built  

### 1. Model Serialization (`train.ipynb`)  

- Trained and evaluated multiple models (Naive Bayes, Logistic Regression, Random Forest) using **scikit-learn**.  
- Selected the best-performing pipeline based on evaluation metrics.  
- Saved the final pipeline to disk as `best_model.joblib` for reusable inference.

---

### 2. Inference Layer (`score.py`)  

Implemented a dedicated `score()` function to abstract prediction logic.

**Function Signature:**

```python
def score(text: str, model, threshold: float) -> tuple[bool, float]
```

**Responsibilities:**

- Computes the probability (propensity) of a message being spam.  
- Applies a configurable threshold to determine the binary prediction.  
- Returns:
  - `prediction` → `bool`
  - `propensity` → `float`

---

### 3. Flask API Deployment (`app.py`)  

- Built a lightweight REST API using **Flask**.  
- Implemented a `/score` endpoint that:
  - Accepts `POST` requests with text input.
  - Loads the serialized model.
  - Returns predictions in JSON format.


---

### 4. Rigorous Automated Testing (`test.py`)  

Developed a comprehensive test suite using **pytest**.

#### Unit Tests  

- **Smoke Test**  
  Ensures the function runs without crashing.

- **Format Test**  
  Validates correct input/output data types.

- **Sanity Checks**  
  - Prediction is strictly boolean (`True` or `False`).  
  - Propensity is bounded between `0.0` and `1.0`.

- **Edge Case Testing**  
  - `threshold = 0.0` → Always predicts spam.  
  - `threshold = 1.0` → Never predicts spam.

- **Typical Input Tests**  
  - Obvious spam messages.  
  - Obvious non-spam (ham) messages.

#### Integration Test  

- Automatically launches the Flask app via `subprocess`.  
- Sends a live HTTP `POST` request to the local `/score` endpoint.  
- Validates the returned JSON response.  
- Gracefully shuts down the server after testing.

---

### 5. Test Coverage (`coverage.txt`)  

- Used **pytest-cov** to measure execution coverage.  
- Achieved **100% test coverage** on the scoring module.

---

## Summary  

This project demonstrates a complete machine learning engineering workflow:

- Model training  
- Model serialization  
- Production-ready inference abstraction  
- REST API deployment  
- Automated unit & integration testing  
- Measurable test coverage  

It bridges the gap between experimentation and production-ready ML systems.
