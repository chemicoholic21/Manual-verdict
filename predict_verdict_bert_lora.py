"""
Prediction script for Verdict classifier using BERT + LoRA model.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import joblib
import os
import argparse

# Column names
JOB_COL = "Grapevine Job - Job → Description"
RESUME_COL = "Grapevine Userresume - Resume → Metadata → Resume Text"
VERDICT_COL = "Verdict"

def load_model(model_dir):
    """
    Load BERT + LoRA model and tokenizer.
    """
    print(f"📂 Loading model from: {model_dir}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label mappings
    label_mappings_path = os.path.join(model_dir, 'label_mappings.joblib')
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"Label mappings not found: {label_mappings_path}")
    
    label_mappings = joblib.load(label_mappings_path)
    num_labels = label_mappings['num_labels']
    id_to_label = label_mappings['id_to_label']
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    print("✅ Model loaded successfully")
    print(f"   Classes: {list(id_to_label.values())}")
    
    return model, tokenizer, id_to_label

def predict_batch(model, tokenizer, texts, batch_size=16, max_length=384):
    """
    Predict verdicts for a batch of texts.
    """
    predictions = []
    probabilities = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        # Move back to CPU
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def predict_csv(model_dir, in_csv, out_csv,
                job_col=JOB_COL,
                resume_col=RESUME_COL,
                verdict_col=VERDICT_COL,
                predict_all=False,
                batch_size=16):
    """
    Predict Verdict for a CSV file using trained BERT + LoRA model.
    """
    # Load model
    model, tokenizer, id_to_label = load_model(model_dir)
    
    # Load CSV
    print(f"📂 Loading CSV from: {in_csv}")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"CSV file not found: {in_csv}")
    
    df = pd.read_csv(in_csv)
    print(f"   Loaded {len(df)} rows")
    
    # Check if required columns exist
    missing_cols = []
    if job_col not in df.columns:
        missing_cols.append(job_col)
    if resume_col not in df.columns:
        missing_cols.append(resume_col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create combined text feature
    df['__text__'] = df[job_col].fillna("").astype(str) + " " + df[resume_col].fillna("").astype(str)
    
    # Determine which rows to predict
    if predict_all:
        mask = pd.Series([True] * len(df))
        print(f"🔮 Predicting Verdict for all {len(df)} rows...")
    else:
        if verdict_col in df.columns:
            mask = df[verdict_col].isna() | (df[verdict_col].astype(str).str.strip() == "")
            print(f"🔮 Predicting Verdict for {mask.sum()} empty rows...")
        else:
            mask = pd.Series([True] * len(df))
            print(f"🔮 Verdict column not found. Predicting for all {len(df)} rows...")
    
    if mask.sum() == 0:
        print("⚠️  No rows to predict (all verdicts are already filled)")
        df.to_csv(out_csv, index=False)
        return
    
    # Get texts to predict
    texts_to_predict = df.loc[mask, '__text__'].values.tolist()
    
    # Predict
    print(f"🤖 Running predictions...")
    predictions, probabilities = predict_batch(model, tokenizer, texts_to_predict, batch_size=batch_size)
    
    # Convert predictions to labels
    predicted_labels = [id_to_label[pred] for pred in predictions]
    
    # Get confidence scores (max probability)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Add confidence column
    df['Verdict_Confidence'] = None
    df.loc[mask, 'Verdict_Confidence'] = confidence_scores
    
    # Update verdict column
    if verdict_col not in df.columns:
        df[verdict_col] = None
    
    # Ensure verdict column is object/string type to avoid dtype warnings
    df[verdict_col] = df[verdict_col].astype('object')
    
    # Convert predictions to string and assign
    df.loc[mask, verdict_col] = [str(label) for label in predicted_labels]
    
    # Save results
    print(f"💾 Saving predictions to: {out_csv}")
    df = df.drop(columns=['__text__'], errors='ignore')
    df.to_csv(out_csv, index=False)
    
    # Print summary
    print(f"\n✅ Predictions complete!")
    print(f"   Total rows: {len(df)}")
    print(f"   Rows predicted: {mask.sum()}")
    if verdict_col in df.columns:
        print(f"\n   Verdict distribution:")
        print(df[verdict_col].value_counts())
    
    avg_confidence = df.loc[mask, 'Verdict_Confidence'].mean()
    print(f"\n   Average prediction confidence: {avg_confidence:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Predict Verdict using BERT + LoRA model')
    parser.add_argument('--model-dir', '-m', required=True,
                       help='Path to trained model directory')
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file to predict on')
    parser.add_argument('--output', '-o', required=True,
                       help='Output CSV file with predictions')
    parser.add_argument('--job-col', default=JOB_COL,
                       help=f'Job description column name (default: {JOB_COL})')
    parser.add_argument('--resume-col', default=RESUME_COL,
                       help=f'Resume text column name (default: {RESUME_COL})')
    parser.add_argument('--verdict-col', default=VERDICT_COL,
                       help=f'Verdict column name (default: {VERDICT_COL})')
    parser.add_argument('--predict-all', action='store_true',
                       help='Predict for all rows, even if Verdict is already filled')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for prediction (default: 16)')
    
    args = parser.parse_args()
    
    predict_csv(
        model_dir=args.model_dir,
        in_csv=args.input,
        out_csv=args.output,
        job_col=args.job_col,
        resume_col=args.resume_col,
        verdict_col=args.verdict_col,
        predict_all=args.predict_all,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()

