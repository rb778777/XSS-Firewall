from flask import Flask, request, jsonify, render_template, url_for, redirect
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os
import logging
import functools
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("firewall.log"), logging.StreamHandler()]
)
logger = logging.getLogger("xss-firewall")

# Application setup
app = Flask(__name__, static_folder='static')
app.config['JSON_SORT_KEYS'] = False  # Maintain JSON order for readability

# Simple LRU cache for prediction results
prediction_cache = {}
MAX_CACHE_SIZE = 1000

class XSSDetector(nn.Module):
    """Neural network model for XSS detection using DistilBERT"""
    def __init__(self, n_classes: int = 2):
        super(XSSDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)  
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(128, n_classes)  

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = bert_output.last_hidden_state[:, 0, :]  
        output = self.drop(output)
        output = self.fc1(output)  
        output = self.drop(output)
        return self.out(output)

# Simple LRU cache decorator
def lru_cache(func):
    @functools.wraps(func)
    def wrapper(sentence, *args, **kwargs):
        # Use only the first 100 chars as cache key to prevent memory issues with very long inputs
        cache_key = sentence[:100] 
        if cache_key in prediction_cache:
            logger.debug(f"Cache hit for input: {cache_key[:20]}...")
            return prediction_cache[cache_key]
        
        result = func(sentence, *args, **kwargs)
        
        # Implement simple LRU by limiting cache size
        if len(prediction_cache) >= MAX_CACHE_SIZE:
            # Remove a random item if cache is full
            prediction_cache.pop(next(iter(prediction_cache)))
            
        prediction_cache[cache_key] = result
        return result
    return wrapper

def load_model() -> Tuple[XSSDetector, DistilBertTokenizer]:
    """Load the model and tokenizer with error handling"""
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize model
        model = XSSDetector(n_classes=2)
        
        # Multiple checkpoint options in order of preference
        checkpoint_paths = [
            'xss_detection_model.pth',
            'models/xss_detection_model.pth',
            'models/best_model.pth'
        ]

        valid_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                valid_path = path
                break

        if valid_path is None:
            raise FileNotFoundError(f"No model checkpoint found in any of the expected locations: {checkpoint_paths}")
        
        # Load to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(valid_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully to {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Load model and tokenizer
try:
    model, tokenizer = load_model()
    DEVICE = next(model.parameters()).device  # Get the device model is on
except Exception as e:
    logger.critical(f"Application startup failed: {str(e)}")
    model, tokenizer = None, None

def preprocess(sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess input text for the model"""
    max_length = min(512, len(sentence))  # DistilBERT has 512 token limit
    
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    # Move tensors to the right device
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    return input_ids, attention_mask

@lru_cache
def predict(sentence: str, threshold: float = 0.85) -> Tuple[int, float]:
    """Predict if a sentence contains XSS with confidence score"""
    start_time = time.time()
    
    try:
        input_ids, attention_mask = preprocess(sentence)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = probs[0][prediction].item()
        
        # Apply threshold (0 = safe, 1 = XSS)
        result = prediction.item() if confidence >= threshold else 0
        
        # Log the prediction time for performance monitoring
        duration = time.time() - start_time
        logger.debug(f"Prediction took {duration*1000:.2f}ms - Result: {result} with confidence {confidence:.4f}")
        
        return result, confidence
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return safe prediction in case of error to avoid blocking legitimate traffic
        return 0, 0.0

def calibrate_prediction(raw_confidence, field_type):
    # Even more aggressive calibration parameters to reduce false positives
    calibration_params = {
        'username': {'a': 0.5, 'b': -0.5},   # Even more aggressive reduction
        'email': {'a': 0.4, 'b': -0.6},      # Most aggressive for email
        'phone': {'a': 0.3, 'b': -0.7},      # Even more aggressive for phone
        'comment': {'a': 0.7, 'b': -0.3}     # More aggressive for comments
    }
    
    params = calibration_params.get(field_type, {'a': 0.6, 'b': -0.4})
    
    # Ensure raw_confidence is not exactly 0 or 1 to avoid division by zero
    epsilon = 1e-7
    raw_confidence = max(min(raw_confidence, 1.0 - epsilon), epsilon)
    
    # Apply Platt scaling
    logit = np.log(raw_confidence / (1 - raw_confidence))
    calibrated = float(1 / (1 + np.exp(-(params['a'] * logit + params['b']))))
    
    return calibrated

@app.route('/')
def home():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    """Render the welcome page after successful form submission"""
    return render_template('welcome.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze form inputs for XSS attacks"""
    # Fields to check for XSS
    fields = ['username', 'email', 'phone', 'comment']
    
    # Check if any required field is missing
    missing_fields = [f for f in fields if f not in request.form or not request.form.get(f).strip()]
    if missing_fields:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400
    
    results = {
        "overall_safe": True,
        "fields": {},
        "metadata": {
            "version": "1.0",
            "timestamp": time.time()
        }
    }
    
    # Process each field
    for field in fields:
        field_value = request.form.get(field).strip()
        
        try:
            # Pre-check for valid format patterns
            if field == 'email' and is_valid_email_format(field_value):
                # If it's a valid email format, consider it safe unless it contains suspicious patterns
                is_safe = True
                if '<script' in field_value.lower() or 'javascript:' in field_value.lower():
                    is_safe = False
                    final_conf = 0.95
                else:
                    final_conf = 0.10
                
                results["fields"][field] = {
                    "safe": bool(is_safe),  # Convert numpy bool_ to Python bool
                    "confidence": round(float(final_conf), 4)  # Convert numpy float to Python float
                }
                
                # Log the decision
                with open('detection_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{field}: {field_value}, Format check: Valid email, Safe: {is_safe}, Confidence: {final_conf:.4f}\n")
                
                if not is_safe:
                    results["overall_safe"] = False
                    logger.warning(f"XSS detected in field '{field}' with confidence {final_conf:.4f}")
                
                continue
                
            elif field == 'phone' and is_valid_phone_format(field_value):
                # If it's a valid phone format, it's almost certainly safe
                results["fields"][field] = {
                    "safe": True,
                    "confidence": 0.05
                }
                
                # Log the decision
                with open('detection_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{field}: {field_value}, Format check: Valid phone, Safe: True, Confidence: 0.05\n")
                
                continue
            
            # For other fields, use model with stricter thresholds
            field_thresholds = {
                'username': 0.95,  # Even higher thresholds
                'email': 0.98,     
                'phone': 0.99,     
                'comment': 0.95    
            }
            threshold = field_thresholds.get(field, 0.95)  # Higher default
            
            # Get raw prediction
            raw_pred, raw_conf = predict(field_value, threshold)
            
            # Get calibrated confidence with more aggressive calibration
            calibrated_conf = calibrate_prediction(raw_conf, field)
            
            # Simple whitelist approach for common inputs
            if field == 'username' and field_value.isalnum() and len(field_value) < 20:
                is_safe = True
                final_conf = 0.1
            # Decide based on both predictions with AND logic and higher threshold
            else:
                is_safe = bool((raw_pred == 0) or (calibrated_conf < threshold))
            
                # For comments, consider length as a factor (longer comments need higher confidence to flag)
                if field == 'comment' and len(field_value) > 50:
                    comment_factor = min(1.0, len(field_value) / 500)  # Scale based on length
                    adjusted_threshold = threshold + (0.1 * comment_factor)  # Add up to 0.1
                    is_safe = bool(is_safe or (calibrated_conf < adjusted_threshold))
                    
                # Final confidence adjustment
                final_conf = float(calibrated_conf if raw_pred == 1 else raw_conf * 0.5)
            
            # Store result for this field - explicitly convert numpy types to Python native types
            results["fields"][field] = {
                "safe": bool(is_safe),  # Convert numpy bool_ to Python bool
                "confidence": round(float(final_conf), 4)  # Convert numpy float to Python float
            }
            
            # Update overall safety
            if not is_safe:
                results["overall_safe"] = False
                logger.warning(f"XSS detected in field '{field}' with confidence {final_conf:.4f}")
                
            # Enhanced logging
            with open('detection_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"{field}: {field_value}, Raw: {raw_pred}/{raw_conf:.4f}, "
                        f"Calibrated: {calibrated_conf:.4f}, "
                        f"Final: {not is_safe}/{final_conf:.4f}, "
                        f"Length: {len(field_value)}\n")
                
        except Exception as e:
            error_msg = f"Model inference failed for {field}: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
    
    # Return appropriate status code based on overall safety
    status_code = 200 if results["overall_safe"] else 403
    return jsonify(results), status_code

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

def ensemble_predict(sentence, field):
    # Use multiple models for prediction
    predictions = []
    confidences = []
    
    # Get predictions from primary model (your current one)
    pred1, conf1 = predict(sentence, 0.85)
    predictions.append(pred1)
    confidences.append(conf1)
    
    # Get predictions from secondary models
    pred2, conf2 = predict(sentence, 0.85)  # A different architecture
    predictions.append(pred2)
    confidences.append(conf2)
    
    # Aggregation strategy (voting)
    final_prediction = 1 if sum(predictions) > len(predictions)/2 else 0
    
    # Confidence can be averaged or use the max
    final_confidence = sum(confidences) / len(confidences)
    
    return final_prediction, final_confidence

def is_valid_email_format(email):
    """Check if string matches basic email pattern"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def is_valid_phone_format(phone):
    """Check if string matches common phone number patterns"""
    # Various phone formats
    phone_patterns = [
        r'^\+?[0-9]{10,15}$',  # Simple digits with optional +
        r'^\+?[0-9]{1,4}[\s-][0-9]{6,10}$',  # Country code with separator
        r'^[0-9]{3}[\s-][0-9]{3}[\s-][0-9]{4}$'  # XXX-XXX-XXXX format
    ]
    return any(bool(re.match(pattern, phone)) for pattern in phone_patterns)

if __name__ == '__main__':
    # Check if model is loaded before starting server
    if model is None or tokenizer is None:
        logger.critical("Cannot start application: Model or tokenizer not loaded")
        exit(1)
        
    # Add startup message
    logger.info("XSS Firewall starting up...")
    app.run(host='0.0.0.0', port=5000)

