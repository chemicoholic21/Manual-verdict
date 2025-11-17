# Verdict Classifier PoC

A proof-of-concept text classifier that predicts "Verdict" (Yes, No, Maybe) based on job descriptions and resume text.

## Features

- **Training**: Trains a TF-IDF + LogisticRegression model on labeled data
- **Prediction**: Predicts Verdict for new CSV files with empty verdict columns
- **Label Cleaning**: Automatically normalizes verdict labels (handles typos like "yse" → "Maybe")
- **Confidence Scores**: Provides prediction confidence when available
- **Flexible**: Works with any CSV that has the required columns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Train on a CSV file with existing Verdict labels:

```bash
python train_verdict_classifier.py \
    --input "Eval 7th October - testabc.csv" \
    --output-model "verdict_classifier_model.joblib" \
    --output-csv "training_data_with_predictions.csv"  # optional
```

**Arguments:**
- `--input` / `-i`: Input CSV file with training data (required)
- `--output-model` / `-o`: Path to save the trained model (default: `verdict_classifier_model.joblib`)
- `--output-csv` / `-c`: Optional CSV output with predictions for all rows

**Example:**
```bash
python train_verdict_classifier.py -i "Eval 7th October - testabc.csv" -o "my_model.joblib"
```

### Step 2: Predict on New Data

Use the trained model to predict Verdict for new CSV files:

```bash
python predict_verdict.py \
    --model "verdict_classifier_model.joblib" \
    --input "new_data.csv" \
    --output "new_data_with_predictions.csv"
```

**Arguments:**
- `--model` / `-m`: Path to trained model file (required)
- `--input` / `-i`: Input CSV file to predict on (required)
- `--output` / `-o`: Output CSV file with predictions (required)
- `--job-col`: Job description column name (default: "Grapevine Job - Job → Description")
- `--resume-col`: Resume text column name (default: "Grapevine Userresume - Resume → Metadata → Resume Text")
- `--verdict-col`: Verdict column name (default: "Verdict")
- `--predict-all`: Predict for all rows, even if Verdict is already filled
- `--confidence-threshold`: Minimum confidence threshold (0.0-1.0)

**Example:**
```bash
python predict_verdict.py \
    -m "verdict_classifier_model.joblib" \
    -i "new_batch.csv" \
    -o "new_batch_with_verdict.csv"
```

## How It Works

1. **Text Feature**: Concatenates job description and resume text into a single text feature
2. **TF-IDF Vectorization**: Converts text to numerical features using Term Frequency-Inverse Document Frequency
3. **Logistic Regression**: Trains a classifier with balanced class weights to handle imbalanced data
4. **Label Cleaning**: Normalizes verdict labels:
   - `yes`, `y`, `advanced` → `Yes`
   - `no`, `n`, `reject` → `No`
   - `maybe`, `may be`, `yse`, `manual intervention` → `Maybe`

## Expected CSV Format

### Training CSV (with Verdict labels):
```csv
Grapevine Job - Job → Description,Grapevine Userresume - Resume → Metadata → Resume Text,Verdict
"Software Engineer role...","John has 5 years...",Yes
"Data Scientist position...","Jane has ML experience...",Maybe
```

### Prediction CSV (with empty Verdict):
```csv
Grapevine Job - Job → Description,Grapevine Userresume - Resume → Metadata → Resume Text,Verdict
"Software Engineer role...","John has 5 years...",
"Data Scientist position...","Jane has ML experience...",
```

## Model Performance

The model will show:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Prediction confidence scores (when using `predict_proba`)

## Limitations & Recommendations

1. **Small/Imbalanced Data**: If your training data is small or lacks examples of certain classes (e.g., no "No" examples), performance may vary. Add more labeled examples for better generalization.

2. **Feature Engineering**: Current PoC uses simple TF-IDF. Consider:
   - Pre-trained transformer encoders (Sentence-BERT)
   - Structured metadata (experience, skills, location)
   - Domain-specific features

3. **Evaluation**: For robust metrics, use k-fold cross-validation on held-out datasets.

4. **Human-in-the-Loop**: Use confidence scores to flag low-confidence predictions for human review.

5. **Production**: Consider wrapping as an API (FastAPI) with logging, versioning, and monitoring.

## Example Workflow

```bash
# 1. Train on labeled data
python train_verdict_classifier.py \
    -i "Eval 7th October - testabc.csv" \
    -o "verdict_model.joblib"

# 2. Predict on new data
python predict_verdict.py \
    -m "verdict_model.joblib" \
    -i "new_candidates.csv" \
    -o "new_candidates_with_verdict.csv"

# 3. Check results
# Open new_candidates_with_verdict.csv to see predictions
```

## Troubleshooting

**Error: "Missing required columns"**
- Ensure your CSV has the correct column names
- Use `--job-col` and `--resume-col` to specify custom column names

**Error: "No valid training data found"**
- Check that your Verdict column has valid labels
- Ensure job description and resume text columns are not empty

**Low prediction confidence**
- Add more training examples
- Ensure training data is representative of prediction data
- Check for data quality issues (empty text, encoding problems)

