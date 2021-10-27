"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import numpy as np
import torch

def de_unfold(x_windows, mask_windows, window_step):
    """
    x_windows of shape (n_windows, n_features, 1, window_size)
    mask_windows of shape (n_windows, n_features, 1, window_size)
    """
    n_windows, n_features, _, window_size = x_windows.shape

    assert (window_step == 1) or (window_step == window_size), 'Window step should be either 1 or equal to window_size'

    len_series = (n_windows)*window_step + (window_size-window_step)

    x = np.zeros((len_series, n_features))
    mask = np.zeros((len_series, n_features))

    n_windows = len(x_windows)
    for i in range(n_windows):
        x_window = x_windows[i,:,0,:]
        x_window = np.swapaxes(x_window,0,1)
        x[i*window_step:(i*window_step+window_size),:] += x_window

        mask_window = mask_windows[i,:,0,:]
        mask_window = np.swapaxes(mask_window,0,1)
        mask[i*window_step:(i*window_step+window_size),:] += mask_window

    division_safe_mask = mask.copy()
    division_safe_mask[division_safe_mask==0]=1
    x = x/division_safe_mask
    mask = 1*(mask>0)
    return x, mask

def kl_multivariate_gaussian(mu, sigma):
    #print('mu.shape', mu.shape)
    #print('sigma.shape', sigma.shape)
    
    D = mu.shape[-1]

    # KL per timestamp
    # assumes that the prior has prior_mu:=0
    # (T, batch, D) KL collapses D
    trace = torch.sum(sigma**2, dim=2)
    mu = torch.sum(mu**2, dim=2)
    log_sigma = torch.sum(torch.log(sigma**2 + 1e-5), dim=2) # torch.log(sigma**2) is the determinant for diagonal + log properties
    kl = 0.5*(trace + mu - log_sigma - D)

    # Mean KL
    kl = torch.mean(kl)

    if torch.isnan(kl):
        print('kl', kl)
        print('trace', trace)
        print('mu', mu)
        print('log_sigma', log_sigma)
        assert 1<0

    return kl


def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def best_f1_linspace(scores, labels, n_splits, segment_adjust):
    best_threshold = 0
    best_f1 = 0
    thresholds = np.linspace(scores.min(),scores.max(), n_splits)
    
    for threshold in thresholds:
        predict = scores>=threshold
        if segment_adjust:
            predict = adjust_predicts(score=scores, label=labels, threshold=None, pred=predict, calc_latency=False)
        f1, *_ = f1_score(predict, labels)

        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
    
    predict = scores>=best_threshold
    if segment_adjust:
        predict = adjust_predicts(score=scores, label=labels, threshold=None, pred=predict, calc_latency=False)
    
    f1, precision, recall, *_ = f1_score(predict, labels)

    return f1, precision, recall, predict, labels, best_threshold


def normalize_scores(scores, interval_size):
    scores_normalized = []
    for score in scores:
        n_intervals = int(np.ceil(len(score)/interval_size))
        score_normalized = []
        for i in range(n_intervals):
            min_timestamp = i*interval_size
            max_timestamp = (i+1)*interval_size
            std = score[:max_timestamp].std()
            score_interval = score[min_timestamp:max_timestamp]/std
            score_normalized.append(score_interval)
        score_normalized =  np.hstack(score_normalized)
        scores_normalized.append(score_normalized)
    return scores_normalized