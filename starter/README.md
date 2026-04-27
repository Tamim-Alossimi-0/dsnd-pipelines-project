# Fashion Forward Forecasting — StyleSense Pipeline Project

## Project Summary

Built a scikit-learn `Pipeline` to predict whether a customer recommends a product based on their review. The dataset contains 18,442 women's clothing reviews from StyleSense with numerical, categorical, and text features.

## Files

- `starter.ipynb` — main notebook with EDA, pipeline construction, training, fine-tuning, and evaluation
- `data/reviews.csv` — anonymized product review dataset (18,442 rows, 9 columns)

## Pipeline Overview

The pipeline uses a `ColumnTransformer` to handle all three data types in a single pass:

| Feature Type | Features | Preprocessing |
|---|---|---|
| Numerical | Age, Positive Feedback Count, Clothing ID | Median imputation → StandardScaler |
| Categorical | Division Name, Department Name, Class Name | Mode imputation → OneHotEncoder |
| Text | Review Text | spaCy lemmatization + stop word removal → TF-IDF (unigrams + bigrams, top 5000 terms) |
| Text | Title | spaCy lemmatization + stop word removal → TF-IDF (top 500 terms) |

**Classifier:** Logistic Regression (C=1.0, max_iter=1000)

**Tuning:** GridSearchCV (cv=3, scoring=F1) over TF-IDF vocab size and regularization strength C

## Results (Test Set — 10% holdout)

| Metric | Score |
|---|---|
| Accuracy | 0.9008 |
| Precision | 0.9254 |
| Recall | 0.9565 |
| F1 Score | 0.9407 |

## Dependencies

```
scikit-learn
numpy
pandas
spacy
```

**Note:** You also need the spaCy English model: `python -m spacy download en_core_web_sm`

Install via: `pip install -r ../requirements.txt`
