# Important imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

#
# Most of the work is based on noobhound's nn solution
#

# NB! Input data files are available in the "../input/" directory when submitting to Kaggle.
input_folder = '../input/'


# Find root mean square logarithmic error for validation
def find_rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


# Preprocessing missing values
def preprocess_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset


# Preprocessing categorical values
def preprocess_categories(train, test):
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)


# Tokenizing raw item descriptions
def preprocess_raw_inputs(train, test):
    raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])

    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)

    train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())


# Padding data for Keras' pleasure
def get_keras_data(dataset):
    X = {'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
         'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
         'brand_name': np.array(dataset.brand_name),
         'category_name': np.array(dataset.category_name),
         'item_condition': np.array(dataset.item_condition_id),
         'num_vars': np.array(dataset[["shipping"]])}
    return X


# Save model states
def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


# Another RMSLE calculation
def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


# Defining RNN model
def create_RNN_model():
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    # RNN layers
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # Main layer
    main_l = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_category_name),
        Flatten()(emb_item_condition),
        rnn_layer1,
        rnn_layer2,
        num_vars])

    main_l = Dropout(0.1)(Dense(128)(main_l))
    main_l = Dropout(0.1)(Dense(64)(main_l))

    # Output layer
    output = Dense(1, activation="linear")(main_l)

    # Compile model
    model = Model([name, item_desc, brand_name, category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model


# Read data and apply preprocessing
train = pd.read_table(input_folder + "train.tsv")
test = pd.read_table(input_folder + "test.tsv")

train = preprocess_missing(train)
test = preprocess_missing(test)

preprocess_categories(train, test)
preprocess_raw_inputs(train, test)


# Analyzing sequences
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))),
                       np.max(test.seq_name.apply(lambda x: len(x)))])

max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x))),
                                   np.max(test.seq_item_description.apply(lambda x: len(x)))])


# Selecting maximal values based on the original work
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([np.max(train.seq_name.max()),
                   np.max(test.seq_name.max()),
                   np.max(train.seq_item_description.max()),
                   np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category_name.max(),
                       test.category_name.max()])+1
MAX_BRAND = np.max([train.brand_name.max(),
                    test.brand_name.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()])+1


# Scaling values
train["target"] = np.log(train.price+1)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
train["target"] = target_scaler.fit_transform(train.target.reshape(-1, 1))
pd.DataFrame(train.target).hist()


# Split training data for crossval
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)


# Pad data for Keras
X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)


# Create model
model = create_RNN_model()
model.summary()


# Teach the model
BATCH_SIZE = 20000
epochs = 5
model.fit(X_train, dtrain.target,
          epochs=epochs,
          batch_size=BATCH_SIZE,
          validation_data=(X_valid, dvalid.target),
          verbose=1)


# Validate model
val_preds = model.predict(X_valid)
val_preds = target_scaler.inverse_transform(val_preds)
val_preds = np.exp(val_preds)+1


# Find the RMSLE
y_true = np.array(dvalid.price.values)
y_pred = val_preds[:, 0]
v_rmsle = find_rmsle(y_true, y_pred)
print("RMSLE: " + str(v_rmsle))


# Test dataset validation
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = target_scaler.inverse_transform(preds)
preds = np.exp(preds)-1


# Result to csv
result = test[["test_id"]]
result["price"] = preds
result.to_csv('submission.csv', index=False)