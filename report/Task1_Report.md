# Explainable Telecom Customer Churn Prediction System Using Machine Learning

**Module:** Advanced Machine Learning (COM 763)
**Assessment:** Portfolio Task 1
**Student Name:** [INSERT STUDENT NAME]
**Student ID:** [INSERT STUDENT ID]
**Word Count:** 1,931 words

## 1. Problem Definition and System Framing

Customer churn—the rate at which customers cancel their subscriptions—represents a critical challenge for the telecommunications industry. High churn directly reduces recurring revenue and inflates customer acquisition costs, as replacing a lost customer is fundamentally more expensive than retaining an existing one (Ullah et al., 2019). This project addresses the business need for early risk identification by developing a predictive machine-learning system. By estimating an individual customer's churn risk from their historical account, service, and billing attributes, the system empowers retention teams to prioritise human-reviewed interventions effectively.

The problem is framed as a supervised binary-classification task, where the target variable, `Churn`, indicates whether a customer left the provider (`Yes` / 1) or remained (`No` / 0). Supervised learning is highly appropriate for this domain because historical patterns of customer attrition are well-documented in the dataset, allowing algorithms to learn the complex associations between features such as tenure or monthly charges and the likelihood of cancellation (Provost and Fawcett, 2013).

The primary intended users of this system are retention specialists, customer-success managers, and business-analytics teams. Consequently, the project aims to deliver not only accurate predictions but also transparent explanations for those predictions. The measurable objectives include establishing a robust data pipeline, comparing candidate classifiers, and deploying an interactive dashboard. Crucially, the system's performance cannot be judged by accuracy alone; due to class imbalance, accuracy may misleadingly obscure poor detection of actual churners (Fawcett, 2006). 

From an ethical and responsible-use perspective, the system is strictly positioned as a decision-support tool. It must not be used for automatic service denial, discrimination, or as a replacement for accountable human review. The fully deployed application, which features prediction, analytics, and explainability capabilities, is accessible at `https://explainable-telecom-customer-churn-prediction.streamlit.app/`.

[INSERT FIGURE 1 HERE: End-to-End Customer Churn Prediction Workflow]

## 2. Data Pipeline and Feature Handling

The foundation of the predictive system relies on the IBM Telco Customer Churn dataset, acquired via Kaggle. The original dataset comprises 7,043 rows and 21 columns, detailing demographics, services, and account information. 

[INSERT TABLE 1 HERE: Dataset and Feature Summary]

A comprehensive data pipeline was established to resolve inherent data-quality issues before model training. The target variable, `Churn`, was encoded into a binary format (1 for `Yes`, 0 for `No`). The dataset exhibits a significant class imbalance, with churners comprising only 26.5% (1,869) of the records. This imbalance is problematic for standard machine-learning algorithms, which often bias towards the majority class to minimise global error rates, resulting in poor minority-class detection (Kuhn and Johnson, 2013). 

[INSERT FIGURE 2 HERE: Churn Distribution Before Resampling]

Data cleaning began with the removal of exact duplicate rows and the stripping of leading and trailing whitespace from all object-type columns. The `SeniorCitizen` attribute, initially provided as an integer (0/1), was mapped to a categorical string ("No"/"Yes") to ensure appropriate subsequent encoding. A notable anomaly was discovered in the `TotalCharges` column, where 11 rows contained blank or whitespace-only strings instead of numeric values. These were forcibly converted to `NaN` and then imputed using the median value (1,397.47) to preserve the dataset size without skewing the distribution. Furthermore, the `CustomerID` column was explicitly dropped from the feature matrix because it carries no predictive power and could lead to spurious associations if inadvertently processed.

To capture nuanced customer behaviour, seven domain-specific features were engineered. For instance, `TenureGroup` bins the continuous `tenure` variable into categories ("New", "Early", "Established", "Loyal"), reflecting distinct lifecycle stages. The `AverageMonthlySpend` was calculated by dividing `TotalCharges` by `tenure` (clipped to a minimum of 1 to prevent division by zero). Additional boolean flags such as `HasSecurityServices`, `HasStreamingServices`, and `HighMonthlyCharge` (using the dataset median as a threshold) were created to simplify complex categorical structures into highly interpretable predictors.

[INSERT FIGURE 3 HERE: Preprocessing and Feature-Engineering Code]

Prior to model training, the dataset was split into training and testing sets. Stratification was employed to guarantee that both sets maintained the original 26.5% churn distribution, preventing anomalous splits (Hastie, Tibshirani and Friedman, 2009). The preprocessing pipeline was constructed using Scikit-learn’s `ColumnTransformer`. Numeric features were subjected to median imputation and standard scaling to ensure uniform magnitude, which is vital for distance-based solvers. Categorical features underwent mode imputation and one-hot encoding, configured to ignore unknown categories during inference (Alkharusi, 2012). 

To resolve the class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was implemented (Chawla et al., 2002). SMOTE generates synthetic minority examples by interpolating between existing churn records. Crucially, SMOTE was embedded within an `imblearn` pipeline to ensure it was applied exclusively to the training folds during cross-validation. Applying SMOTE to the entire dataset prior to splitting would cause severe data leakage, artificially inflating validation scores by evaluating the model on synthetic data derived from the test set (Fernández et al., 2018).

[INSERT FIGURE 4 HERE: Class Balance After SMOTE]

## 3. Model Implementation and Debugging

The experimental design required comparing a simple, interpretable baseline model against more complex, non-linear tree-based ensembles. The selected candidate algorithms were Logistic Regression, Decision Tree, Random Forest, and XGBoost. Logistic Regression provides a strong baseline with high global interpretability through its coefficients. Decision Trees offer transparent rule-based logic, though they are prone to overfitting. Random Forest and XGBoost (Chen and Guestrin, 2016) were included as powerful ensemble techniques capable of capturing complex, non-linear feature interactions that simpler models might miss.

To guarantee reproducibility and prevent data leakage, all preprocessing steps, SMOTE balancing, and estimator logic were encapsulated within a unified `imblearn.pipeline.Pipeline`. This design ensures that during both cross-validation and inference, raw data passes through exactly the same sequence of transformations. All models were instantiated with a fixed `random_state` and, where applicable, configured with `class_weight="balanced"` to further penalise minority-class misclassification. Logistic Regression was configured with the `lbfgs` solver and 1,000 maximum iterations to ensure convergence, whilst the Decision Tree depth was restricted to 8 to prevent extreme overfitting. The Random Forest was instantiated with 200 estimators and a maximum depth of 12, and XGBoost used a learning rate of 0.1 with `eval_metric="logloss"`.

[INSERT FIGURE 5 HERE: Model Training and Cross-Validation Code]

The models were evaluated using stratified 5-fold cross-validation. This technique partitions the training data into five distinct subsets, iteratively training on four and validating on the fifth, providing a robust estimate of generalisation performance while mitigating variance (Hastie, Tibshirani and Friedman, 2009). 

Throughout the implementation, several critical debugging interventions were required. Initial testing revealed that the `clean_dataset` function correctly dropped duplicate rows, which inadvertently caused index mismatches in early test-suite designs. By refining the `pytest` architecture to account for deduplication, the integrity of the data-cleaning phase was validated. Furthermore, during local testing of the Streamlit interface, it became evident that the application would fail gracefully if the model artefacts or dataset were absent. To ensure the cloud deployment supported the in-app "Model Training" page, the original Kaggle CSV dataset (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`) was explicitly removed from the `.gitignore` exclusions and committed to the Git repository. This resolved a deployment issue where Streamlit Community Cloud lacked the necessary training data, effectively enabling remote pipeline execution. 

Additionally, ensuring SHAP (SHapley Additive exPlanations) compatibility with transformed features required careful alignment. Because Scikit-learn’s `OneHotEncoder` alters the feature space dimensionality, a custom helper function (`get_feature_names_from_pipeline`) was implemented to extract the precise post-transformation feature names, allowing SHAP summary plots to label the input variables correctly.

## 4. Experimental Evaluation and Model Selection

Evaluating predictive models on imbalanced datasets requires metrics that look beyond standard accuracy. Because non-churners dominate the dataset, a naïve model predicting "No Churn" for every customer would achieve approximately 73.5% accuracy but possess zero practical value (Fawcett, 2006). Therefore, the primary selection criteria prioritised the F1-score, which computes the harmonic mean of precision and recall. Recall is particularly vital in telecom retention, as it measures the proportion of actual churners the model successfully identifies. ROC-AUC was used as a secondary metric to assess the model's general discriminatory ability.

The cross-validation and final held-out test results for the four candidate models are summarised below.

**Table 2. Model Comparison Results**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV F1 Mean | CV F1 Std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7417 | 0.5086 | 0.7888 | 0.6184 | 0.8396 | 0.6299 | 0.0253 |
| Decision Tree | 0.7615 | 0.5411 | 0.6684 | 0.5981 | 0.8025 | 0.5868 | 0.0214 |
| Random Forest | 0.7757 | 0.5668 | 0.6578 | 0.6089 | 0.8337 | 0.6182 | 0.0179 |
| XGBoost | 0.7771 | 0.5820 | 0.5695 | 0.5757 | 0.8321 | 0.5939 | 0.0327 |

*Note: Extract from `outputs/model_comparison.csv` generated by `src/evaluation.py`.*

The results indicate an association between algorithmic complexity and class-balance sensitivity. While XGBoost achieved the highest overall accuracy (0.7771) and precision (0.5820), it struggled with recall (0.5695). Random Forest performed steadily with an F1-score of 0.6089. However, Logistic Regression demonstrated superior minority-class detection, achieving the highest F1-score (0.6184), the highest ROC-AUC (0.8396), and a vastly superior recall of 0.7888. 

The final model selection rationale utilised a composite heuristic (60% F1, 20% Recall, 20% ROC-AUC). Logistic Regression was explicitly chosen as the final model because it successfully identified nearly 79% of all churning customers. In a business context, the operational cost of a false positive (offering a retention discount to a customer who was not going to leave) is generally much lower than the cost of a false negative (losing a high-value customer because the model failed to identify them) (Provost and Fawcett, 2013). Furthermore, Logistic Regression is inherently more interpretable, allowing stakeholders to trust the decision-making process.

[INSERT FIGURE 6 HERE: Final Model Confusion Matrix]
[INSERT FIGURE 7 HERE: Final Model ROC Curve]

To provide explainability, SHAP values were computed to measure the marginal contribution of each feature (Lundberg and Lee, 2017). The global SHAP summary plot revealed that `tenure`, `InternetService_Fiber optic`, and `MonthlyCharges` were the strongest drivers of model output. However, it must be stated that SHAP values represent statistical associations learned by the model, not causal proof of churn (Molnar, 2022). 

[INSERT FIGURE 8 HERE: SHAP Global Feature Importance]

## 5. Deployment, Limitations and Conclusion

The complete machine-learning pipeline is deployed as an interactive web application on Streamlit Community Cloud. The application facilitates several user views: a dynamic Prediction Page for scoring new customer inputs, Dataset Analytics for exploratory data analysis, Model Performance for reviewing evaluation metrics, and an Explainability view detailing local and global SHAP insights. The repository is available at `https://github.com/johndoe553/Customer-Churn-Prediction`.

[INSERT FIGURE 9 HERE: Streamlit Customer Prediction Page]
[INSERT FIGURE 10 HERE: Streamlit Analytics, Performance, or Explainability Page]

Despite its success, the system exhibits several academic and practical limitations. The sample dataset may not accurately represent all modern telecom providers or geographical regions, and historical data may encapsulate outdated biases. Over time, the model will inevitably suffer from concept drift as consumer behaviour and market pricing change; thus, continuous monitoring and periodic retraining are strictly required. Furthermore, the model's false positives and false negatives carry real operational consequences. 

In conclusion, the project successfully achieved its aim. By engineering robust features, correctly implementing SMOTE inside cross-validation, and selecting a model optimised for recall, the resulting system effectively identifies at-risk customers while remaining transparent. The application serves as a highly functional, responsible decision-support tool that augments—but never replaces—human retention strategy.
