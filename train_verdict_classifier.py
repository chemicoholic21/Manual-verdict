"""
Training script for Verdict classifier PoC.
Trains a TF-IDF + LogisticRegression model on job descriptions and resume text.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import os

# Column names
JOB_COL = "Grapevine Job - Job → Description"
RESUME_COL = "Grapevine Userresume - Resume → Metadata → Resume Text"
VERDICT_COL = "Verdict"

def clean_verdict_label(label):
    """
    Clean and normalize Verdict labels.
    Maps variations to standard: Yes, No, Maybe
    Blank/empty cells are treated as "No"
    """
    # Handle NaN and empty strings as "No"
    if pd.isna(label):
        return 'No'
    
    label_str = str(label).strip().lower()
    
    # Handle empty strings as "No"
    if label_str == '' or label_str == 'nan':
        return 'No'
    
    # Map variations to standard labels
    if label_str in ['yes', 'y', 'advanced']:
        return 'Yes'
    elif label_str in ['no', 'n', 'reject']:
        return 'No'
    elif label_str in ['maybe', 'may be', 'yse', 'manual intervention', 'manual']:
        return 'Maybe'
    else:
        # Default to No for unknown labels (changed from Maybe)
        return 'No'

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
        # Where Verdict is empty/NaN and Result[LLM] is "Reject", set Verdict to "No"
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
    
    # Filter out rows with empty text (but keep all verdicts, including "No" from blanks)
    mask = (df['__text__'].str.strip() != "")
    df_filtered = df[mask].copy()
    
    print(f"\n✅ After filtering: {len(df_filtered)} rows with valid labels and text")
    
    if len(df_filtered) == 0:
        raise ValueError("No valid training data found!")
    
    X = df_filtered['__text__'].values
    y = df_filtered['__verdict_cleaned__'].values
    
    return X, y, df_filtered

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train TF-IDF + LogisticRegression pipeline.
    """
    print(f"\n🚀 Training model...")
    print(f"   Training samples: {len(X)}")
    print(f"   Classes: {np.unique(y)}")
    
    # Check if we have enough samples for train/test split
    if len(X) < 10:
        print("⚠️  Warning: Very small dataset, using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        # Stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # If stratification fails (e.g., one class has only 1 sample), use regular split
            print("⚠️  Warning: Stratified split failed, using regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    
    # Calculate custom class weights to give more importance to Yes and Maybe
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Boost Yes and Maybe classes even more
    if 'Yes' in class_weight_dict:
        class_weight_dict['Yes'] *= 2.0  # Give Yes 2x more weight
    if 'Maybe' in class_weight_dict:
        class_weight_dict['Maybe'] *= 1.5  # Give Maybe 1.5x more weight
    
    print(f"   Class weights: {class_weight_dict}")
    
    # Create pipeline: TF-IDF + LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,  # ignore terms that appear in less than 2 documents
            max_df=0.95,  # ignore terms that appear in more than 95% of documents
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight=class_weight_dict  # use custom weights
        ))
    ])
    
    # Train
    print(f"   Training on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    if len(X_test) > 0:
        y_pred = pipeline.predict(X_test)
        print(f"\n📈 Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    return pipeline

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Verdict classifier')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with training data')
    parser.add_argument('--output-model', '-o', default='verdict_classifier_model.joblib',
                       help='Output path for trained model (default: verdict_classifier_model.joblib)')
    parser.add_argument('--output-csv', '-c', default=None,
                       help='Output CSV with predictions (optional)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from: {args.input}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")
    
    df = pd.read_csv(args.input)
    
    # Prepare training data
    X, y, df_filtered = prepare_training_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    print(f"\n💾 Saving model to: {args.output_model}")
    joblib.dump(model, args.output_model)
    print("✅ Model saved successfully!")
    
    # Optionally save CSV with predictions
    if args.output_csv:
        print(f"\n📝 Generating predictions for all rows...")
        df['__text__'] = df[JOB_COL].fillna("").astype(str) + " " + df[RESUME_COL].fillna("").astype(str)
        
        # Predict for all rows
        predictions = model.predict(df['__text__'].values)
        df['Verdict_Predicted'] = predictions
        
        # Fill empty verdicts with predictions
        if VERDICT_COL in df.columns:
            mask = df[VERDICT_COL].isna() | (df[VERDICT_COL].astype(str).str.strip() == "")
            df.loc[mask, VERDICT_COL] = predictions[mask]
        
        df.to_csv(args.output_csv, index=False)
        print(f"✅ Predictions saved to: {args.output_csv}")
    
    print(f"\n🎉 Training complete!")
    print(f"   Model: {args.output_model}")
    print(f"   Use this model with predict_verdict.py to predict on new CSVs")

if __name__ == "__main__":
    main()

