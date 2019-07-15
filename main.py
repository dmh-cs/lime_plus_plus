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

from lime_plus_plus.rule_explainer import RuleExplainer
from lime_plus_plus.runs import adult


def main():
  print('evaluating on adult dataset')
  df, feature_names, cat_mapping, target_names = adult()
  feature_names = list(cat_mapping.keys()) + list(set(feature_names) - set(cat_mapping.keys()))
  df = df[feature_names + ['target']]
  df.target = df.target.apply(lambda x: '>' in x)
  categorical_features = [df.columns.tolist().index(name) for name in cat_mapping.keys()]
  train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(df.values[:, :-1].astype(float),
                                                                                    df.target.values.astype(int),
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
                              target_names=target_names,
                              max_rule_len=max_rule_len,
                              categorical_features=categorical_features,
                              from_training_set=from_training_set,
                              one_hot=one_hot,
                              cat_mapping=cat_mapping)

    # to_explain = test[np.random.randint(0, test.shape[0])]
    to_explain = test[3]
    model = explainer.train_local_tree(to_explain,
                                       rf.predict_proba,
                                       num_samples=10000)
    explainer.explain(rf.predict_proba, model, to_explain)
