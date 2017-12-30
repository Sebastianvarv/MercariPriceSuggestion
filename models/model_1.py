# Important imports
import eli5
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error

# NB! Input data files are available in the "../input/" directory.

#
# Most of the work is based on K.Lopuhin's kernel.
#

# NB! Input data files are available in the "../input/" directory when submitting to Kaggle.
input_folder = '../data/'


# Preprocessing
def preprocess(dataset):
    dataset['category_name'] = dataset['category_name'].fillna('Other').astype(str)
    dataset['brand_name'] = dataset['brand_name'].fillna('missing').astype(str)
    dataset['shipping'] = dataset['shipping'].astype(str)
    dataset['item_condition_id'] = dataset['item_condition_id'].astype(str)
    dataset['item_description'] = dataset['item_description'].fillna('None')
    return dataset


# Preprocessing and feature extraction.
def preprocess_and_extract(dataset):
    y_dataset = np.log1p(dataset['price'])
    dataset = dataset.drop('price', 1)

    # Slight preprocessing
    dataset = preprocess(dataset)

    # Feature extraction.
    default_preprocessor = CountVectorizer().build_preprocessor()
    def build_preprocessor(field):
        field_idx = list(dataset.columns).index(field)
        return lambda x: default_preprocessor(x[field_idx])

    vectorizer = FeatureUnion([
        ('name', CountVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            preprocessor=build_preprocessor('name'))),
        ('category_name', CountVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('category_name'))),
        ('brand_name', CountVectorizer(
            token_pattern='.+',
            preprocessor=build_preprocessor('brand_name'))),
        ('shipping', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('shipping'))),
        ('item_condition_id', CountVectorizer(
            token_pattern='\d+',
            preprocessor=build_preprocessor('item_condition_id'))),
        ('item_description', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=100000,
            preprocessor=build_preprocessor('item_description'))),
    ])
    X_dataset = vectorizer.fit_transform(dataset.values)

    return X_dataset, y_dataset, vectorizer


def extract_test_features(dataset, vectorizer):
    X_dataset = vectorizer.transform(dataset.values)
    return X_dataset


# Find root mean square logarithmic error for validation.
def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


# Read training data and apply minor preprocessing and feature extraction.
train = pd.read_table(input_folder + 'train.tsv')
X_train, y_train, train_vectorizer = preprocess_and_extract(train)


# Make a Ridge model and run k-fold validation.
cv = KFold(n_splits=10, shuffle=True, random_state=42)
model = None

for train_ids, valid_ids in cv.split(X_train):
    model = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=0.5,
        max_iter=100,
        normalize=False,
        tol=0.05)

    model.fit(X_train[train_ids], y_train[train_ids])
    y_pred_valid = model.predict(X_train[valid_ids])
    rmsle = get_rmsle(y_pred_valid, y_train[valid_ids])
    print(f'valid rmsle: {rmsle:.5f}')
    break


# Predict on test set.
test = pd.read_table(input_folder + 'test.tsv')
X_test = extract_test_features(test, train_vectorizer)

test_ids = test['test_id'].values
y_pred_test = model.predict(X_test[test_ids])
print(y_pred_test)

result = pd.DataFrame(
    {'test_id': test_ids,
     'price': y_pred_test
    })

result.to_csv('submission.csv', index=False)