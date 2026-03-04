def score(text: str, model, threshold: float) -> tuple[bool, float]:
    """
    Scores a trained model on a given text.
    """
    # Predict probabilities. predict_proba returns a 2D array.
    # Index 1 is the probability of the positive class (spam).
    propensity = float(model.predict_proba([text])[0][1])
    
    # Generate a boolean prediction based on the threshold
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity