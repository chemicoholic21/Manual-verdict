"""
Prediction script for Verdict classifier PoC.
Uses a trained model to predict Verdict for new CSV files.
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse

# Column names
JOB_COL = "Grapevine Job - Job → Description"
RESUME_COL = "Grapevine Userresume - Resume → Metadata → Resume Text"
VERDICT_COL = "Verdict"

def predict_csv(model_path, in_csv, out_csv,
                job_col=JOB_COL,
                resume_col=RESUME_COL,
                verdict_col=VERDICT_COL,
                predict_all=False,
                confidence_threshold=0.0):
    """
    Predict Verdict for a CSV file using a trained model.
    
    Args:
        model_path: Path to the trained .joblib model file
        in_csv: Input CSV file path
        out_csv: Output CSV file path
        job_col: Name of job description column
        resume_col: Name of resume text column
        verdict_col: Name of verdict column
        predict_all: If True, predict for all rows. If False, only predict for empty verdicts.
        confidence_threshold: Minimum confidence to accept prediction (0.0-1.0)
    """
    # Load model
    print(f"📂 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
    
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
    
    # Get predictions
    texts_to_predict = df.loc[mask, '__text__'].values
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(texts_to_predict)
        classes = model.classes_
        
        # Use threshold-based prediction for better balance
        # If Yes probability > 0.25, predict Yes
        # If Maybe probability > 0.15, predict Maybe
        # Otherwise predict No
        predictions = []
        for probs in probabilities:
            prob_dict = dict(zip(classes, probs))
            
            if 'Yes' in prob_dict and prob_dict['Yes'] > 0.25:
                predictions.append('Yes')
            elif 'Maybe' in prob_dict and prob_dict['Maybe'] > 0.15:
                predictions.append('Maybe')
            else:
                predictions.append('No')
        
        predictions = np.array(predictions)
        max_probs = np.max(probabilities, axis=1)
        
        # Add confidence column
        df['Verdict_Confidence'] = None
        df.loc[mask, 'Verdict_Confidence'] = max_probs
        
        # Filter by confidence threshold if specified
        if confidence_threshold > 0:
            low_confidence_mask = max_probs < confidence_threshold
            low_confidence_count = low_confidence_mask.sum()
            if low_confidence_count > 0:
                print(f"⚠️  {low_confidence_count} predictions have confidence < {confidence_threshold}")
                print(f"   These will still be included but marked for review")
    else:
        # Fallback to standard prediction
        predictions = model.predict(texts_to_predict)
    
    # Update verdict column
    if verdict_col not in df.columns:
        df[verdict_col] = None
    
    # Convert predictions to string to avoid dtype issues
    df.loc[mask, verdict_col] = predictions.astype(str)
    
    # Save results
    print(f"💾 Saving predictions to: {out_csv}")
    df = df.drop(columns=['__text__'], errors='ignore')  # Remove temporary column
    df.to_csv(out_csv, index=False)
    
    # Print summary
    print(f"\n✅ Predictions complete!")
    print(f"   Total rows: {len(df)}")
    print(f"   Rows predicted: {mask.sum()}")
    if verdict_col in df.columns:
        print(f"\n   Verdict distribution:")
        print(df[verdict_col].value_counts())
    
    if 'Verdict_Confidence' in df.columns:
        avg_confidence = df.loc[mask, 'Verdict_Confidence'].mean()
        print(f"\n   Average prediction confidence: {avg_confidence:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Predict Verdict using trained model')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained model file (.joblib)')
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
    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Minimum confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    predict_csv(
        model_path=args.model,
        in_csv=args.input,
        out_csv=args.output,
        job_col=args.job_col,
        resume_col=args.resume_col,
        verdict_col=args.verdict_col,
        predict_all=args.predict_all,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main()

