import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger("xss-train")

# Default configuration
CONFIG = {
    "dataset_path": "XSS_dataset.csv",
    "model_dir": "models",
    "model_name": "xss_detection_model.pth",
    "max_len": 256,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "dropout": 0.3,
    "test_size": 0.1,
    "random_seed": 42,
    "use_amp": True,  # Automatic Mixed Precision
    "save_checkpoints": True,
    "checkpoint_interval": 1
}

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class XSSDataset(Dataset):
    """Dataset for XSS detection"""
    def __init__(self, sentences: List[str], labels: List[int], tokenizer, max_len: int):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, item) -> Dict[str, Any]:
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class XSSDetector(nn.Module):
    """Neural network model for XSS detection"""
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super(XSSDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = bert_output.last_hidden_state[:, 0, :]
        output = self.drop(output)
        output = self.fc1(output)
        output = self.relu(output)
        return self.out(output)

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load and validate CSV dataset"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['Sentence', 'Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
        # Check datatypes
        if not pd.api.types.is_string_dtype(df['Sentence']):
            logger.warning("'Sentence' column is not string type, converting...")
            df['Sentence'] = df['Sentence'].astype(str)
            
        if not pd.api.types.is_numeric_dtype(df['Label']):
            raise ValueError("'Label' column must contain numeric values (0 or 1)")
        
        # Check if binary classification (0 and 1)
        unique_labels = df['Label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Labels must be binary (0 or 1), found: {unique_labels}")
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Dataset contains missing values. Handling...")
            df['Sentence'].fillna('', inplace=True)
            df.dropna(inplace=True)
            
        # Data statistics
        logger.info(f"Dataset loaded: {len(df)} records")
        label_counts = df['Label'].value_counts()
        logger.info(f"Class distribution: {dict(label_counts)}")
        
        # Sample counts for each class
        class_0_percent = (label_counts.get(0, 0) / len(df)) * 100
        class_1_percent = (label_counts.get(1, 0) / len(df)) * 100
        logger.info(f"Class 0 (Safe): {label_counts.get(0, 0)} samples ({class_0_percent:.2f}%)")
        logger.info(f"Class 1 (XSS): {label_counts.get(1, 0)} samples ({class_1_percent:.2f}%)")
        
        # Extracting data
        sentences = df['Sentence'].to_numpy()
        labels = df['Label'].to_numpy()
        
        return df, sentences, labels
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples, scaler, use_amp=True):
    """Train the model for one epoch"""
    model.train()
    losses = []
    correct_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training Batches")
    for batch_idx, d in enumerate(progress_bar):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        optimizer.zero_grad()

        if use_amp and torch.cuda.is_available():
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses[-100:]) if losses else 0,
            'acc': (correct_predictions.double() / ((progress_bar.n + 1) * data_loader.batch_size)).item()
        })

        # Clear GPU cache periodically during training
        if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """Evaluate the model"""
    model.eval()
    losses = []
    correct_predictions = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Validation Batches"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create classification report
    report = classification_report(all_labels, all_preds, target_names=['Safe', 'XSS'], output_dict=True)
    
    metrics = {
        'accuracy': correct_predictions.double().item() / n_examples,
        'loss': np.mean(losses),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    return metrics

def save_model(model, path, metrics=None):
    """Save the model and optionally metrics"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")
    
    # Save metrics if provided
    if metrics:
        metrics_path = path.replace('.pth', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

def plot_training_metrics(training_history, output_dir='plots'):
    """Plot training and validation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    epochs = len(training_history['train_acc'])
    x_axis = list(range(1, epochs + 1))
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, training_history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(x_axis, training_history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_plot.png", dpi=300)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, training_history['train_loss'], 'b-', label='Training Loss')
    plt.plot(x_axis, training_history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_plot.png", dpi=300)
    
    # F1 Score plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, training_history['val_f1'], 'g-', label='F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/f1_score_plot.png", dpi=300)
    
    logger.info(f"Training plots saved to {output_dir}")

def main(config):
    """Main training function"""
    set_seed(config['random_seed'])
    
    # Create model directory
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['model_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Load and validate dataset
    df, sentences, labels = load_and_validate_data(config['dataset_path'])
    
    # Split data
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=config['test_size'], 
        random_state=config['random_seed'], stratify=labels
    )
    
    logger.info(f"Training set: {len(train_sentences)} samples")
    logger.info(f"Validation set: {len(val_sentences)} samples")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = XSSDataset(train_sentences, train_labels, tokenizer, config['max_len'])
    val_dataset = XSSDataset(val_sentences, val_labels, tokenizer, config['max_len'])
    
    # Create data loaders
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Use 0 to disable multiprocessing
        pin_memory=True
    )
    
    val_data_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,  # Match the training loader
        pin_memory=True
    )
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Print CUDA device information
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA Device {i}: {device_name}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Calculate class weights for imbalanced dataset
    label_counts = df['Label'].value_counts().to_dict()
    n_samples = len(df)
    class_weights = torch.tensor(
        [n_samples / (label_counts.get(0, 1) * 2), 
         n_samples / (label_counts.get(1, 1) * 2)], 
        dtype=torch.float32, 
        device=device
    )
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Initialize model
    model = XSSDetector(n_classes=2, dropout=config['dropout']).to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Initialize loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize learning rate scheduler
    total_steps = len(train_data_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Initialize mixed precision scaler
    if config['use_amp'] and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        # If CUDA is not available, disable AMP
        config['use_amp'] = False
        logger.info("Mixed precision training (AMP) disabled because CUDA is not available")
    
    # Training history
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'val_f1': []
    }
    
    # Training loop
    best_val_f1 = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        logger.info("=" * 30)
        
        # Train
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, 
            len(train_dataset), scaler, use_amp=config['use_amp']
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        val_metrics = eval_model(model, val_data_loader, loss_fn, device, len(val_dataset))
        val_acc = val_metrics['accuracy']
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1']
        
        # Update history
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save checkpoint if requested
        if config['save_checkpoints'] and (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config['model_dir'], 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_model(model, checkpoint_path, val_metrics)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(config['model_dir'], "best_model.pth")
            save_model(model, best_model_path, val_metrics)
            logger.info(f"New best model saved with F1 score: {val_f1:.4f}")
        
        # Clear memory between epochs
        clear_gpu_memory()
    
    # Save final model
    final_model_path = os.path.join(config['model_dir'], config['model_name'])
    save_model(model, final_model_path, val_metrics)
    
    # Plot training metrics
    plot_training_metrics(history)
    
    # Calculate and log training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Log best validation metrics
    best_epoch = np.argmax(history['val_f1']) + 1
    logger.info(f"Best model at epoch {best_epoch} with F1 score: {best_val_f1:.4f}")
    
    if torch.cuda.is_available() and config['batch_size'] > 16:
        config['learning_rate'] *= (config['batch_size'] / 16)  # Scale learning rate with batch size
        logger.info(f"Scaled learning rate to {config['learning_rate']} for larger batch size")
    
    # Use mixed precision only on newer GPU architectures
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        config['use_amp'] = True
    else:
        config['use_amp'] = False
        logger.info("Mixed precision disabled due to older GPU architecture")
    
    return final_model_path

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Display PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train XSS detection model')
    parser.add_argument('--config', type=str, help='Path to config file (JSON)')
    args = parser.parse_args()
    
    # Load configuration
    config = CONFIG
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
            
    # Run training
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
        
    # Set appropriate batch size for RTX 3050 (4GB VRAM)
    config['batch_size'] = 16
        
    model_path = main(config)
    logger.info(f"Final model saved to {model_path}")
    
