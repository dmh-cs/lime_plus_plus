from random import shuffle
from itertools import product
from collections import Counter

import pydash as _
import numpy as np
import pandas as pd
from sklearn import tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import graphviz

class RuleExplainer():
  def __init__(self, train_data, feature_names, class_names, max_rule_len, categorical_features, from_training_set, one_hot, kernel_width=None):
    self.train_data, self.feature_names, self.class_names, self.max_rule_len, self.categorical_features, self.from_training_set, self.one_hot = train_data, feature_names, class_names, max_rule_len, categorical_features, from_training_set, one_hot
    self.scaler = sklearn.preprocessing.StandardScaler()
    self.scaler.fit(train_data)
    self.feature_values = {}
    self.feature_frequencies = {}
    for feature in self.categorical_features:
      column = training_data[:, feature]
      feature_count = Counter(column)
      values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
      self.feature_values[feature] = values
      self.feature_frequencies[feature] = (np.array(frequencies) /
                                           float(sum(frequencies)))
      self.scaler.mean_[feature] = 0
      self.scaler.scale_[feature] = 1
    if kernel_width is None:
      kernel_width = np.sqrt(self.one_hot.transform(training_data).shape[1]) * .75
      # kernel_width = np.sqrt(training_data.shape[1]) * .75
    kernel_width = float(kernel_width)
    def kernel(d):
      return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2)).reshape(-1)
    self.kernel_fn = kernel

  def train_local_tree(self, row, model_pred_proba, num_samples):
    if self.from_training_set not in [None, 0.0]:
      num_from_train = int(self.from_training_set * num_samples)
      num_random_samples = num_samples - num_from_train
      training_idxs = list(range(len(training_data))); shuffle(training_idxs)
      dataset_samples = np.stack([training_data[idx] for idx in training_idxs[:num_from_train]])
    else:
      num_from_train = 0
      num_random_samples = num_samples
    perturbed = row + np.random.normal(0, 1, (num_random_samples, len(row))) * self.scaler.scale_
    for column in self.categorical_features:
      values = self.feature_values[column]
      freqs = self.feature_frequencies[column]
      perturbed[:, column] = np.random.choice(values, size=num_random_samples, replace=True, p=freqs)
    if num_from_train > 0:
      local_data = np.concatenate([perturbed, dataset_samples], 0)
    else:
      local_data = perturbed
    trans = self.one_hot.transform(local_data)
    distances = sklearn.metrics.pairwise_distances(
      one_hot.transform(self.scaler.transform(local_data)),
      one_hot.transform(self.scaler.transform(row.reshape(1, -1))),
      metric='euclidean'
    ).ravel()
    weights = self.kernel_fn(distances)
    self.trans = trans
    self.distances = distances
    self.weights = weights
    labels = model_pred_proba(trans)
    model_regressor = DecisionTreeRegressor(max_depth=self.max_rule_len - 1)
    model_regressor.fit(trans, labels[:, 1], sample_weight=weights / weights.sum())
    return model_regressor


def eval_faithfulness(weights, samples, model, surrogate):
  model_preds = model(samples) > 0.5
  surrogate_preds = surrogate(samples) > 0.5
  return np.mean(weights * (model_preds == surrogate_preds))

feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
cat_mapping = {
  'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
  'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
  'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
  'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
  'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
  'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
  'sex': ['Female', 'Male'],
  'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
}

adult = pd.read_csv('./data/adult.data', header=None, names=feature_names + ['target'])
for col_name, options in cat_mapping.items():
  adult[col_name] = adult[col_name].apply(lambda x: _.index_of(options, x.strip()))
  adult[col_name] = adult[col_name].apply(lambda x: x if (x != -1) else max(adult[col_name]) + 1)
feature_names = list(cat_mapping.keys()) + list(set(feature_names) - set(cat_mapping.keys()))
adult = adult[feature_names + ['target']]
target_names = ['<=50K', '>50K']
adult.target = adult.target.apply(lambda x: '>' in x)
categorical_features = [adult.columns.tolist().index(name) for name in cat_mapping.keys()]
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(adult.values[:, :-1].astype(float),
                                                                                  adult.target.values.astype(int),
                                                                                  train_size=0.80)
one_hot = OneHotEncoder(sparse=False, handle_unknown='ignore', categorical_features=categorical_features).fit(train)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(one_hot.transform(train), labels_train)
print('acc', sklearn.metrics.accuracy_score(labels_test, rf.predict(one_hot.transform(test))))

# for max_rule_len, from_training_set in product([2, 3, 4], [0.2, 0.4, 0.6, 0.8]):
for max_rule_len, from_training_set in product([5], [0.0, 0.5, 0.8, 0.9]):
  np.random.seed(1234)
  print('max_rule_len: {}, from_training_set: {}'.format(max_rule_len, from_training_set))
  explainer = RuleExplainer(train,
                            feature_names=feature_names,
                            class_names=target_names,
                            max_rule_len=max_rule_len,
                            categorical_features=categorical_features,
                            from_training_set=from_training_set,
                            one_hot=one_hot)

  # to_explain = test[np.random.randint(0, test.shape[0])]
  to_explain = test[3]
  prediction = rf.predict(one_hot.transform(to_explain.reshape(1, -1)))
  training_data = train
  model = explainer.train_local_tree(to_explain,
                                     rf.predict_proba,
                                     num_samples=10000)
  path = model.decision_path(one_hot.transform(to_explain.reshape(1, -1)))
  final_node_id = model.apply(one_hot.transform(to_explain.reshape(1, -1)))
  feature = model.tree_.feature
  threshold = model.tree_.threshold

  sample_id = 0
  node_index = path.indices[path.indptr[sample_id]:
                            path.indptr[sample_id + 1]]

  print('Rule faithfulness (weighted acc): {}'.format(eval_faithfulness(explainer.weights, explainer.trans, lambda x: rf.predict_proba(x)[:, 1], model.predict)))

  print('Rules used to predict sample %s: with prediction %s' % (sample_id, target_names[prediction[0]]))
  for node_id in node_index:
    if final_node_id[sample_id] == node_id:
      continue

    feature_idx = feature[node_id]
    name_idx = np.searchsorted(one_hot.feature_indices_[1:], feature_idx)
    is_cat = name_idx < len(cat_mapping)
    if is_cat:
      feature_value = cat_mapping[feature_names[name_idx]][int(to_explain[name_idx])]
    else:
      name_idx = feature_idx - max(one_hot.feature_indices_) + len(cat_mapping)
      feature_value = to_explain[name_idx]
    if (one_hot.transform(to_explain.reshape(1, -1))[0, feature[node_id]] <= threshold[node_id]):
      threshold_sign = "<="
    else:
      threshold_sign = ">"

    if is_cat:
      print("%s = %s"
        % (feature_names[name_idx],
           feature_value))
    else:
      print("%s = %s %s %s"
          % (feature_names[name_idx],
             feature_value,
             threshold_sign,
             threshold[node_id]))
