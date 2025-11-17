# Verdict Classifier System - One Pager

## Overview
An AI-powered text classification system that automatically predicts candidate verdicts (Yes/No/Maybe) based on job descriptions and resume text. The system uses machine learning to streamline candidate evaluation and reduce manual review time.

## How It Works
1. **Training**: Learns patterns from labeled data (job descriptions + resumes → verdicts)
2. **Text Processing**: Combines job description and resume text into a single feature
3. **Classification**: Uses TF-IDF vectorization + Logistic Regression to predict verdicts
4. **Output**: Generates predictions with confidence scores for each candidate

## Key Features
- ✅ **Automatic Verdict Prediction**: Classifies candidates as Yes, No, or Maybe
- ✅ **Smart Label Mapping**: Automatically maps "Reject" from Result[LLM] to "No"
- ✅ **Handles Missing Data**: Treats blank verdict cells as "No" during training
- ✅ **Confidence Scores**: Provides prediction confidence for quality control
- ✅ **Threshold-Based Logic**: Balanced predictions using probability thresholds
- ✅ **Class Balancing**: Custom weights to handle imbalanced training data

## Usage

### Training
```bash
python3 train_verdict_classifier.py \
    -i "training_data.csv" \
    -o "verdict_classifier_model.joblib"
```

### Prediction
```bash
python3 predict_verdict.py \
    -m "verdict_classifier_model.joblib" \
    -i "candidates.csv" \
    -o "predictions.csv"
```

## Technical Specifications

**Model Architecture:**
- **Feature Extraction**: TF-IDF Vectorizer (5,000 max features, unigrams + bigrams)
- **Classifier**: Logistic Regression with balanced class weights
- **Classes**: Yes, No, Maybe
- **Model Size**: ~137 KB

**Training Data Requirements:**
- Required columns: "Grapevine Job - Job → Description", "Grapevine Userresume - Resume → Metadata → Resume Text", "Verdict"
- Optional: "Result[LLM]" column (maps "Reject" → "No")
- Label format: Yes, No, Maybe (case-insensitive)

**Prediction Logic:**
- Yes: Predicted if probability > 25%
- Maybe: Predicted if probability > 15% (and Yes < 25%)
- No: Default prediction

## Performance Metrics

**Current Model (trained on 501 samples):**
- **Accuracy**: 87%
- **Class Distribution**: No (91%), Yes (7%), Maybe (2%)
- **Average Confidence**: 68-73%

**Training Data:**
- Total samples: 501
- No: 455 (239 from "Reject" mapping + 216 from blanks)
- Yes: 36
- Maybe: 10

## Output Format

The prediction CSV includes:
- All original columns
- **Verdict**: Predicted label (Yes/No/Maybe)
- **Verdict_Confidence**: Prediction confidence score (0-1)

## Benefits

1. **Time Savings**: Automatically processes hundreds of candidates in seconds
2. **Consistency**: Standardized evaluation criteria across all candidates
3. **Scalability**: Handles large candidate pools efficiently
4. **Quality Control**: Confidence scores help identify edge cases for review
5. **Flexibility**: Easy to retrain with new data for improved accuracy

## Limitations & Recommendations

- **Small Training Set**: Current model trained on 501 samples; more data improves accuracy
- **Class Imbalance**: Limited "Yes" and "Maybe" examples; add more for better balance
- **Feature Engineering**: Currently uses simple text features; could be enhanced with:
  - Pre-trained embeddings (Sentence-BERT)
  - Structured metadata (experience, skills, location)
  - Domain-specific features

## Next Steps

1. **Expand Training Data**: Add more labeled examples, especially "Yes" and "Maybe"
2. **Feature Enhancement**: Integrate structured candidate metadata
3. **Model Improvement**: Experiment with transformer-based embeddings
4. **Production Deployment**: Wrap as API service for real-time predictions
5. **Monitoring**: Track prediction accuracy and model drift over time

---

**Contact**: For questions or improvements, refer to `VERDICT_CLASSIFIER_README.md` for detailed documentation.

