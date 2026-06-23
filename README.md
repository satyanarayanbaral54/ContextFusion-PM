# ContextFuse-PM — Contextual Predictive Maintenance

## Problem
Most predictive maintenance models rely only on internal sensor signals and
fail in real-world environments. ContextFuse-PM fuses IoT telemetry with
external context (weather, factory load) to improve failure prediction.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)
10,000 rows | 6 sensor features | ~3.4% failure rate

## Results
| Configuration       | Macro F1 |
|---------------------|----------|
| IoT signals only    | X.XXXX   |
| IoT + External      | X.XXXX   |  ← improvement

## Pipeline
1. Ingestion → 2. Rolling Features → 3. Context Fusion →
4. SMOTE + LightGBM (5-fold CV) → 5. SHAP → 6. Noise Testing

## Usage
pip install -r requirements.txt
python src/ingestion.py
python src/generate_external.py
python src/features.py
python src/fusion.py
python src/model.py
python src/evaluate.py
python src/noise_test.py
