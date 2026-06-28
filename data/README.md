# Dataset setup

This repository includes the required CSV dataset to enable reproducible training and deployed Streamlit analytics:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Source attribution: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) on Kaggle.

4. From the repository root, regenerate the deployment artefacts:

   ```bash
   python scripts/train_model.py
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The app's prediction pages, dataset analytics, and in-app retraining use the tracked CSV file.

Dataset attribution: IBM sample data, republished by BlastChar on Kaggle. The dataset is governed separately by the source authors' rights and Kaggle's terms.
