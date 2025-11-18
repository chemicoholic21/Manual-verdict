"""
Training script for Verdict classifier using BERT with LoRA fine-tuning.
Uses transformer-based embeddings with parameter-efficient fine-tuning.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import argparse
from typing import List, Dict

# Column names
JOB_COL = "Grapevine Job - Job → Description"
RESUME_COL = "Grapevine Userresume - Resume → Metadata → Resume Text"
VERDICT_COL = "Verdict"

# Model configuration
MODEL_NAME = "distilbert-base-uncased"  # Lightweight BERT model
MAX_LENGTH = 384  # Maximum sequence length (optimal for DistilBERT)
BATCH_SIZE = 16
LEARNING_RATE = 5e-5  # Lower LR for LoRA fine-tuning
NUM_EPOCHS = 10

def clean_verdict_label(label):
    """
    Clean and normalize Verdict labels.
    Maps variations to standard: Yes, No, Maybe
    Blank/empty cells are treated as "No"
    """
    if pd.isna(label):
        return 'No'
    
    label_str = str(label).strip().lower()
    
    if label_str == '' or label_str == 'nan':
        return 'No'
    
    if label_str in ['yes', 'y', 'advanced']:
        return 'Yes'
    elif label_str in ['no', 'n', 'reject']:
        return 'No'
    elif label_str in ['maybe', 'may be', 'yse', 'manual intervention', 'manual']:
        return 'Maybe'
    else:
        return 'No'

class VerdictDataset(Dataset):
    """Dataset class for verdict classification."""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label to id mapping
        unique_labels = sorted(set(labels))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_to_id[label], dtype=torch.long)
        }

def prepare_training_data(df, job_col=JOB_COL, resume_col=RESUME_COL, verdict_col=VERDICT_COL):
    """
    Prepare training data by concatenating job and resume text,
    and cleaning verdict labels.
    """
    print(f"📊 Input CSV shape: {df.shape}")
    
    # Check if required columns exist
    missing_cols = []
    if job_col not in df.columns:
        missing_cols.append(job_col)
    if resume_col not in df.columns:
        missing_cols.append(resume_col)
    if verdict_col not in df.columns:
        missing_cols.append(verdict_col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create combined text feature
    df['__text__'] = df[job_col].fillna("").astype(str) + " " + df[resume_col].fillna("").astype(str)
    
    # Map Result[LLM] column to Verdict if Verdict is empty
    result_col = "Result[LLM]"
    if result_col in df.columns:
        print(f"\n🔍 Mapping Result[LLM] to Verdict...")
        mask_reject = (df[verdict_col].isna() | (df[verdict_col].astype(str).str.strip() == "")) & \
                      (df[result_col].astype(str).str.strip().str.lower() == "reject")
        df.loc[mask_reject, verdict_col] = "No"
        print(f"   Mapped {mask_reject.sum()} 'Reject' values from Result[LLM] to 'No' in Verdict")
    
    # Clean verdict labels
    print(f"\n🔍 Cleaning Verdict labels...")
    print(f"Raw Verdict value counts:")
    print(df[verdict_col].value_counts())
    
    df['__verdict_cleaned__'] = df[verdict_col].apply(clean_verdict_label)
    
    print(f"\nCleaned Verdict value counts:")
    print(df['__verdict_cleaned__'].value_counts())
    
    # Filter out rows with empty text
    mask = (df['__text__'].str.strip() != "")
    df_filtered = df[mask].copy()
    
    print(f"\n✅ After filtering: {len(df_filtered)} rows with valid labels and text")
    
    if len(df_filtered) == 0:
        raise ValueError("No valid training data found!")
    
    X = df_filtered['__text__'].values.tolist()
    y = df_filtered['__verdict_cleaned__'].values.tolist()
    
    return X, y, df_filtered

def setup_lora_model(model_name: str, num_labels: int):
    """
    Setup BERT model with LoRA configuration.
    """
    print(f"\n🔧 Setting up BERT model with LoRA...")
    print(f"   Model: {model_name}")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Configure LoRA
    # For DistilBERT, target all attention linear layers for better performance
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,  # LoRA rank (reduced for small datasets, prevents overfitting)
        lora_alpha=16,  # LoRA alpha (typically 2x rank)
        lora_dropout=0.05,  # Reduced dropout
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # All attention layers
        bias="none"  # Don't train bias terms
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

def train_model(X, y, model_name=MODEL_NAME, test_size=0.2, random_state=42, 
                num_epochs=None, batch_size=None, learning_rate=None, output_dir='./bert_lora_verdict_model'):
    """
    Train BERT model with LoRA fine-tuning.
    """
    print(f"\n🚀 Training BERT model with LoRA...")
    print(f"   Training samples: {len(X)}")
    print(f"   Classes: {sorted(set(y))}")
    
    # Get unique labels
    unique_labels = sorted(set(y))
    num_labels = len(unique_labels)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    # Load tokenizer
    print(f"   Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split data - avoid data leakage for small datasets
    if len(X) < 10:
        print("⚠️  Warning: Very small dataset, using 80/20 split without stratification")
        # Force 80/20 split to avoid leakage
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            print("⚠️  Warning: Stratified split failed, using regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    
    # Create datasets
    train_dataset = VerdictDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = VerdictDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Setup model with LoRA
    model = setup_lora_model(model_name, num_labels)
    
    # Detect device (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        use_gradient_checkpointing = True
        use_fp16 = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        use_gradient_checkpointing = False  # MPS doesn't support gradient checkpointing well
        use_fp16 = False  # MPS doesn't support fp16
        print("   🍎 Using MPS (Apple Silicon) - gradient checkpointing disabled")
    else:
        device = torch.device('cpu')
        use_gradient_checkpointing = False  # CPU doesn't benefit much
        use_fp16 = False
    
    # Enable gradient checkpointing only if supported
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")
    
    # Use provided parameters or defaults
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    batch = batch_size if batch_size is not None else BATCH_SIZE
    lr = learning_rate if learning_rate is not None else LEARNING_RATE
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    import torch.nn as nn
    
    class_weights = compute_class_weight('balanced', classes=np.array(unique_labels), y=y_train)
    class_weight_dict = dict(zip(unique_labels, class_weights))
    # Create tensor with weights in label order
    class_weight_tensor = torch.tensor([class_weight_dict[id_to_label[i]] for i in range(num_labels)], dtype=torch.float32)
    
    print(f"   Class weights: {class_weight_dict}")
    
    # Move class weights to device
    class_weight_tensor = class_weight_tensor.to(device)
    
    def weighted_loss(predictions, labels):
        loss_fct = nn.CrossEntropyLoss(weight=class_weight_tensor)
        return loss_fct(predictions, labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_fp16,  # Use mixed precision if supported
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        warmup_ratio=0.1,  # 10% warmup steps
        gradient_checkpointing=use_gradient_checkpointing,  # Save memory (disabled for MPS)
    )
    
    # Create trainer with custom loss
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # Note: HuggingFace Trainer doesn't directly support custom loss in Trainer
        # We'll handle class imbalance through class weights in the model
    )
    
    # Override compute_loss to use weighted loss
    original_compute_loss = trainer.compute_loss
    
    def compute_loss(model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Move labels to same device as model
        if labels is not None:
            labels = labels.to(device)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weight_tensor)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    trainer.compute_loss = compute_loss
    
    # Train
    print(f"\n📚 Starting training...")
    trainer.train()
    
    # Evaluate
    print(f"\n📈 Model Performance:")
    eval_results = trainer.evaluate()
    print(f"   Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    # Use saved label_to_id mapping directly (more reliable)
    y_true = [label_to_id[label] for label in y_test]
    
    # Convert back to labels
    y_pred_labels = [id_to_label[pred] for pred in y_pred]
    y_true_labels = y_test
    
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels))
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    label_mappings = {
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_labels': num_labels
    }
    joblib.dump(label_mappings, os.path.join(output_dir, 'label_mappings.joblib'))
    
    return model, tokenizer, label_mappings

def main():
    parser = argparse.ArgumentParser(description='Train Verdict classifier with BERT + LoRA')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with training data')
    parser.add_argument('--output-dir', '-o', default='./bert_lora_verdict_model',
                       help='Output directory for trained model')
    parser.add_argument('--model-name', default=MODEL_NAME,
                       help=f'Base model name (default: {MODEL_NAME})')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from: {args.input}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")
    
    df = pd.read_csv(args.input)
    
    # Prepare training data
    X, y, df_filtered = prepare_training_data(df)
    
    # Train model with provided parameters
    model, tokenizer, label_mappings = train_model(
        X, y, 
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    print(f"\n💾 Model saved to: {args.output_dir}")
    print(f"✅ Training complete!")
    print(f"   Use predict_verdict_bert_lora.py to predict on new CSVs")

if __name__ == "__main__":
    main()

