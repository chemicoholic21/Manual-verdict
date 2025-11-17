#!/bin/bash
# Example usage script for Verdict Classifier PoC

echo "🚀 Verdict Classifier PoC - Example Usage"
echo ""

# Check if training file exists
TRAINING_FILE="Eval 7th October - testabc.csv"
MODEL_FILE="verdict_classifier_model.joblib"

if [ ! -f "$TRAINING_FILE" ]; then
    echo "⚠️  Training file '$TRAINING_FILE' not found."
    echo "   Please ensure your training CSV is in the current directory."
    echo ""
fi

echo "Step 1: Train the model"
echo "python train_verdict_classifier.py -i \"$TRAINING_FILE\" -o \"$MODEL_FILE\""
echo ""

echo "Step 2: Predict on new data"
echo "python predict_verdict.py -m \"$MODEL_FILE\" -i \"new_data.csv\" -o \"new_data_with_predictions.csv\""
echo ""

echo "For detailed instructions, see VERDICT_CLASSIFIER_README.md"

