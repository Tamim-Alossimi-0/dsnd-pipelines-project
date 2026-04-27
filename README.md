# Fashion Forward Forecasting — StyleSense Pipeline Project

Predict whether a customer recommends a product based on their review text, age, product category, and other features. Built as part of the Udacity Data Scientist Nanodegree.

## Getting Started

The main notebook is in `starter/starter.ipynb`. The dataset is in `starter/data/reviews.csv`.

### Dependencies

```
scikit-learn
numpy
pandas
spacy
```

### Installation

1. Install the required packages:

```
pip install -r requirements.txt
```

2. Download the spaCy English model:

```
python -m spacy download en_core_web_sm
```

3. Open the notebook:

```
jupyter notebook starter/starter.ipynb
```

## Project Overview

The dataset contains 18,442 anonymized women's clothing reviews with 8 features (numerical, categorical, and text) and a binary target (`Recommended IND`).

The pipeline uses `ColumnTransformer` to handle all data types:

- **Numerical** (Age, Positive Feedback Count, Clothing ID): median imputation + standard scaling
- **Categorical** (Division Name, Department Name, Class Name): mode imputation + one-hot encoding
- **Text** (Review Text, Title): spaCy lemmatization and stop word removal + TF-IDF vectorization

The classifier is Logistic Regression, tuned with GridSearchCV (3-fold CV, optimized for F1 score).

## Built With

- [scikit-learn](https://scikit-learn.org/) - ML pipeline and model training
- [spaCy](https://spacy.io/) - NLP text preprocessing (tokenization, lemmatization)
- [pandas](https://pandas.pydata.org/) - Data loading and manipulation
- [NumPy](https://numpy.org/) - Numerical operations

## License

[License](LICENSE.txt)
