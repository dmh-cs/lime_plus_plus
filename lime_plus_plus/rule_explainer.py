import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)
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

from .evaluation import eval_faithfulness

class RuleExplainer():
  def __init__(self, train_data, feature_names, target_names, max_rule_len, categorical_features, from_training_set, one_hot, cat_mapping, kernel_width=None):
    self.train_data, self.feature_names, self.max_rule_len, self.categorical_features, self.from_training_set, self.one_hot = train_data, feature_names, max_rule_len, categorical_features, from_training_set, one_hot
    self.target_names = target_names
    self.cat_mapping = cat_mapping
    self.scaler = sklearn.preprocessing.StandardScaler()
    self.scaler.fit(train_data)
    self.feature_values = {}
    self.feature_frequencies = {}
    for feature in self.categorical_features:
      column = self.train_data[:, feature]
      feature_count = Counter(column)
      values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
      self.feature_values[feature] = values
      self.feature_frequencies[feature] = (np.array(frequencies) /
                                           float(sum(frequencies)))
      self.scaler.mean_[feature] = 0
      self.scaler.scale_[feature] = 1
    if kernel_width is None:
      kernel_width = np.sqrt(self.one_hot.transform(self.train_data).shape[1]) * .75
    kernel_width = float(kernel_width)
    def kernel(d):
      return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2)).reshape(-1)
    self.kernel_fn = kernel

  def train_local_tree(self, row, model_pred_proba, num_samples):
    if self.from_training_set not in [None, 0.0]:
      num_from_train = int(self.from_training_set * num_samples)
      num_random_samples = num_samples - num_from_train
      training_idxs = list(range(len(self.train_data))); shuffle(training_idxs)
      dataset_samples = np.stack([self.train_data[idx] for idx in training_idxs[:num_from_train]])
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
      self.one_hot.transform(self.scaler.transform(local_data)),
      self.one_hot.transform(self.scaler.transform(row.reshape(1, -1))),
      metric='euclidean'
    ).ravel()
    weights = self.kernel_fn(distances)
    self.trans = trans
    self.distances = distances
    self.weights = weights
    labels = model_pred_proba(trans)
    local_dt = DecisionTreeRegressor(max_depth=self.max_rule_len - 1)
    local_dt.fit(trans, labels[:, 1], sample_weight=weights / weights.sum())
    return local_dt

  def explain(self, model_pred_proba, local_dt, to_explain):
    prediction = int(model_pred_proba(self.one_hot.transform(to_explain.reshape(1, -1)))[:, 1] > 0.5)
    path = local_dt.decision_path(self.one_hot.transform(to_explain.reshape(1, -1)))
    final_node_id = local_dt.apply(self.one_hot.transform(to_explain.reshape(1, -1)))
    feature = local_dt.tree_.feature
    threshold = local_dt.tree_.threshold

    sample_id = 0
    node_index = path.indices[path.indptr[sample_id]:
                              path.indptr[sample_id + 1]]

    print('Rule faithfulness (weighted acc): {}'.format(eval_faithfulness(self.weights, self.trans, lambda x: model_pred_proba(x)[:, 1], local_dt.predict)))
    print('Rules used to predict sample %s: with prediction %s' % (sample_id, self.target_names[prediction]))
    for node_id in node_index:
      if final_node_id[sample_id] == node_id:
        continue

      feature_idx = feature[node_id]
      name_idx = np.searchsorted(self.one_hot.feature_indices_[1:], feature_idx)
      is_cat = name_idx < len(self.cat_mapping)
      if is_cat:
        feature_value = self.cat_mapping[self.feature_names[name_idx]][int(to_explain[name_idx])]
      else:
        name_idx = feature_idx - max(self.one_hot.feature_indices_) + len(self.cat_mapping)
        feature_value = to_explain[name_idx]
      if (self.one_hot.transform(to_explain.reshape(1, -1))[0, feature[node_id]] <= threshold[node_id]):
        threshold_sign = "<="
      else:
        threshold_sign = ">"

      if is_cat:
        print("%s = %s"
          % (self.feature_names[name_idx],
             feature_value))
      else:
        print("%s = %s %s %s"
            % (self.feature_names[name_idx],
               feature_value,
               threshold_sign,
               threshold[node_id]))
