import numpy as np

def eval_faithfulness(weights, samples, model, surrogate):
  model_preds = model(samples) > 0.5
  surrogate_preds = surrogate(samples) > 0.5
  return np.mean(weights * (model_preds == surrogate_preds))
