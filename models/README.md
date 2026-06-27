# Model artefacts

These small generated artefacts are intentionally versioned so the prediction and explainability features work immediately on Streamlit Community Cloud. They contain fitted parameters, transformed feature names, aggregate metrics, and a transformed SHAP background sample—not raw dataset rows or credentials.

Regenerate them after downloading the dataset:

```bash
python scripts/train_model.py
```
