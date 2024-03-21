from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

def input_fn_train(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


def input_fn_predict(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


CSV_COLUMN_NAMES = ['id', 'date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer', 'bc_price_evo']
TEST_CSV_COLUMN_NAMES = ['id', 'date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
RES = ['DOWN', 'UP']
class_mapping = {'DOWN': 0, 'UP': 1}

train_path = "./train.csv"
test_path = "./test.csv"

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=TEST_CSV_COLUMN_NAMES, header=0)

train_y = train.pop('bc_price_evo')
train_y = train_y.map(class_mapping)
train.pop('id')
test_ids = test['id']
test.pop('id')

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=2)

classifier.train(
    input_fn=lambda: input_fn_train(train, train_y, training=True),
    steps=5000)

predictions = classifier.predict(
    input_fn=lambda: input_fn_predict(test))

results = []

for pred_dict, id in zip(predictions, test_ids):
    class_id = pred_dict['class_ids'][0]
    results.append((id, RES[class_id]))

# Convert the results to a DataFrame
results_df = pd.DataFrame(results, columns=['id', 'bc_price_evo'])

# Write the results to a CSV file
results_df.to_csv('predictions.csv', index=False)