# Dataset setup

The dataset is intentionally not versioned because its Kaggle data card does not state a clear licence permitting public redistribution.

1. Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle using your own account and acceptance of its terms.
2. Extract the archive.
3. Copy the CSV to this exact path:

   ```text
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

4. From the repository root, regenerate the deployment artefacts:

   ```bash
   python scripts/train_model.py
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The app's prediction pages use the committed model artefacts when the CSV is absent. Dataset analytics and retraining require the CSV and display a helpful missing-data message otherwise.

Dataset attribution: IBM sample data, republished by BlastChar on Kaggle. The dataset is governed separately by the source authors' rights and Kaggle's terms.
