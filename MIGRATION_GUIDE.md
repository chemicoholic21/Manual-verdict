# Migration Guide: TF-IDF → BERT + LoRA

## Complete System Rewrite

The verdict classifier has been completely rewritten to use **BERT with LoRA** instead of TF-IDF + Logistic Regression.

## What Changed

### Architecture

**Old System:**
```
Text → TF-IDF Vectorizer → Logistic Regression → Verdict
```

**New System:**
```
Text → BERT Tokenizer → DistilBERT Encoder → LoRA Adapters → Classification Head → Verdict
```

### Files

| Old File | New File | Status |
|----------|----------|--------|
| `train_verdict_classifier.py` | `train_verdict_bert_lora.py` | ✅ New |
| `predict_verdict.py` | `predict_verdict_bert_lora.py` | ✅ New |
| `verdict_classifier_model.joblib` | `bert_lora_verdict_model/` | ✅ New format |

**Old files are preserved** - you can still use them if needed.

## Key Improvements

1. **Semantic Understanding**: BERT understands meaning, not just keywords
2. **Better Accuracy**: Expected 90%+ vs. 87% with TF-IDF
3. **Context-Aware**: Handles synonyms and related terms
4. **Efficient Training**: LoRA trains only ~1% of parameters
5. **GPU Support**: Automatic GPU detection and usage

## Migration Steps

### Step 1: Install New Dependencies

```bash
pip install torch transformers peft accelerate datasets
```

Or update all:
```bash
pip install -r requirements.txt
```

### Step 2: Retrain Model

```bash
python3 train_verdict_bert_lora.py \
    -i "Eval 7th October - MetaForms Senior AI.csv" \
    -o "./bert_lora_verdict_model"
```

**Note:** Training takes longer (~5-10 minutes vs. seconds) but produces better results.

### Step 3: Update Predictions

```bash
python3 predict_verdict_bert_lora.py \
    -m "./bert_lora_verdict_model" \
    -i "tf front posteval.csv" \
    -o "tf_front_posteval_with_predictions.csv"
```

## Command Comparison

### Training

**Old:**
```bash
python3 train_verdict_classifier.py -i "data.csv" -o "model.joblib"
```

**New:**
```bash
python3 train_verdict_bert_lora.py -i "data.csv" -o "./bert_lora_verdict_model"
```

### Prediction

**Old:**
```bash
python3 predict_verdict.py -m "model.joblib" -i "test.csv" -o "out.csv"
```

**New:**
```bash
python3 predict_verdict_bert_lora.py -m "./bert_lora_verdict_model" -i "test.csv" -o "out.csv"
```

## Model Format

**Old:** Single `.joblib` file (~137 KB)
```
verdict_classifier_model.joblib
```

**New:** Directory with multiple files (~500 MB)
```
bert_lora_verdict_model/
├── adapter_config.json
├── adapter_model.bin
├── config.json
├── label_mappings.joblib
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

## Performance Comparison

| Metric | TF-IDF | BERT + LoRA |
|--------|--------|-------------|
| Training Time | ~5 seconds | ~5-10 minutes |
| Prediction Time | ~1 second/100 rows | ~1-2 seconds/100 rows |
| Accuracy | 87% | Expected 90%+ |
| Memory | Low (~100 MB) | Medium (~2 GB) |
| GPU Required | No | Recommended |
| Model Size | 137 KB | ~500 MB |

## When to Use Each

**Use TF-IDF (Old System) if:**
- Need fast training/prediction
- Limited computational resources
- Simple keyword matching is sufficient
- Working with very small datasets

**Use BERT + LoRA (New System) if:**
- Need highest accuracy
- Have GPU available
- Want semantic understanding
- Working with larger datasets
- Need context-aware predictions

## Backward Compatibility

- ✅ Old TF-IDF models still work
- ✅ Old prediction scripts still work
- ✅ Both systems can coexist
- ✅ You can compare results from both

## Troubleshooting

**CUDA/GPU Issues:**
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU: The code automatically falls back to CPU

**Memory Issues:**
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Reduce max length in training script (change `MAX_LENGTH = 256`)

**Import Errors:**
- Ensure all dependencies: `pip install -r requirements.txt`
- Check PyTorch: `python3 -c "import torch; print(torch.__version__)"`

## Next Steps

1. **Train new model** on your data
2. **Compare results** with old TF-IDF model
3. **Choose best system** for your use case
4. **Update workflows** to use new system if preferred

## Support

- **Documentation**: See `BERT_LORA_README.md`
- **Quick Start**: See `BERT_LORA_QUICKSTART.md`
- **Old System**: Still available in original files

