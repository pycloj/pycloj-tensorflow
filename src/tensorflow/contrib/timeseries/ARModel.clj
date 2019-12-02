(ns tensorflow.contrib.timeseries.ARModel
  "Auto-regressive model, both linear and non-linear.

  Features to the model include time and values of input_window_size timesteps,
  and times for output_window_size timesteps. These are passed through a
  configurable prediction model, and then fed to a loss function (e.g. squared
  loss).

  Note that this class can also be used to regress against time only by setting
  the input_window_size to zero.

  Each periodicity in the `periodicities` arg is divided by the
  `num_time_buckets` into time buckets that are represented as features added
  to the model.

  A good heuristic for picking an appropriate periodicity for a given data set
  would be the length of cycles in the data. For example, energy usage in a
  home is typically cyclic each day. If the time feature in a home energy
  usage dataset is in the unit of hours, then 24 would be an appropriate
  periodicity. Similarly, a good heuristic for `num_time_buckets` is how often
  the data is expected to change within the cycle. For the aforementioned home
  energy usage dataset and periodicity of 24, then 48 would be a reasonable
  value if usage is expected to change every half hour.

  Each feature's value for a given example with time t is the difference
  between t and the start of the time bucket it falls under. If it doesn't fall
  under a feature's associated time bucket, then that feature's value is zero.

  For example: if `periodicities` = (9, 12) and `num_time_buckets` = 3, then 6
  features would be added to the model, 3 for periodicity 9 and 3 for
  periodicity 12.

  For an example data point where t = 17:
  - It's in the 3rd time bucket for periodicity 9 (2nd period is 9-18 and 3rd
    time bucket is 15-18)
  - It's in the 2nd time bucket for periodicity 12 (2nd period is 12-24 and
    2nd time bucket is between 16-20).

  Therefore the 6 added features for this row with t = 17 would be:

  # Feature name (periodicity#_timebucket#), feature value
  P9_T1, 0 # not in first time bucket
  P9_T2, 0 # not in second time bucket
  P9_T3, 2 # 17 - 15 since 15 is the start of the 3rd time bucket
  P12_T1, 0 # not in first time bucket
  P12_T2, 1 # 17 - 16 since 16 is the start of the 2nd time bucket
  P12_T3, 0 # not in third time bucket
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce timeseries (import-module "tensorflow.contrib.timeseries"))

(defn ARModel 
  "Auto-regressive model, both linear and non-linear.

  Features to the model include time and values of input_window_size timesteps,
  and times for output_window_size timesteps. These are passed through a
  configurable prediction model, and then fed to a loss function (e.g. squared
  loss).

  Note that this class can also be used to regress against time only by setting
  the input_window_size to zero.

  Each periodicity in the `periodicities` arg is divided by the
  `num_time_buckets` into time buckets that are represented as features added
  to the model.

  A good heuristic for picking an appropriate periodicity for a given data set
  would be the length of cycles in the data. For example, energy usage in a
  home is typically cyclic each day. If the time feature in a home energy
  usage dataset is in the unit of hours, then 24 would be an appropriate
  periodicity. Similarly, a good heuristic for `num_time_buckets` is how often
  the data is expected to change within the cycle. For the aforementioned home
  energy usage dataset and periodicity of 24, then 48 would be a reasonable
  value if usage is expected to change every half hour.

  Each feature's value for a given example with time t is the difference
  between t and the start of the time bucket it falls under. If it doesn't fall
  under a feature's associated time bucket, then that feature's value is zero.

  For example: if `periodicities` = (9, 12) and `num_time_buckets` = 3, then 6
  features would be added to the model, 3 for periodicity 9 and 3 for
  periodicity 12.

  For an example data point where t = 17:
  - It's in the 3rd time bucket for periodicity 9 (2nd period is 9-18 and 3rd
    time bucket is 15-18)
  - It's in the 2nd time bucket for periodicity 12 (2nd period is 12-24 and
    2nd time bucket is between 16-20).

  Therefore the 6 added features for this row with t = 17 would be:

  # Feature name (periodicity#_timebucket#), feature value
  P9_T1, 0 # not in first time bucket
  P9_T2, 0 # not in second time bucket
  P9_T3, 2 # 17 - 15 since 15 is the start of the 3rd time bucket
  P12_T1, 0 # not in first time bucket
  P12_T2, 1 # 17 - 16 since 16 is the start of the 2nd time bucket
  P12_T3, 0 # not in third time bucket
  "
  [periodicities input_window_size output_window_size num_features & {:keys [prediction_model_factory num_time_buckets loss exogenous_feature_columns]
                       :or {exogenous_feature_columns None}} ]
    (py/call-attr-kw timeseries "ARModel" [periodicities input_window_size output_window_size num_features] {:prediction_model_factory prediction_model_factory :num_time_buckets num_time_buckets :loss loss :exogenous_feature_columns exogenous_feature_columns }))

(defn define-loss 
  "Default loss definition with state replicated across a batch.

    Time series passed to this model have a batch dimension, and each series in
    a batch can be operated on in parallel. This loss definition assumes that
    each element of the batch represents an independent sample conditioned on
    the same initial state (i.e. it is simply replicated across the batch). A
    batch size of one provides sequential operations on a single time series.

    More complex processing may operate instead on get_start_state() and
    get_batch_loss() directly.

    Args:
      features: A dictionary (such as is produced by a chunker) with at minimum
        the following key/value pairs (others corresponding to the
        `exogenous_feature_columns` argument to `__init__` may be included
        representing exogenous regressors):
        TrainEvalFeatures.TIMES: A [batch size x window size] integer Tensor
            with times for each observation. If there is no artificial chunking,
            the window size is simply the length of the time series.
        TrainEvalFeatures.VALUES: A [batch size x window size x num features]
            Tensor with values for each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL). For INFER,
        see predict().
    Returns:
      A ModelOutputs object.
    "
  [ self features mode ]
  (py/call-attr self "define_loss"  self features mode ))

(defn exogenous-feature-columns 
  "`tf.feature_colum`s for features which are not predicted."
  [ self ]
    (py/call-attr self "exogenous_feature_columns"))

(defn generate 
  ""
  [ self number_of_series series_length model_parameters seed ]
  (py/call-attr self "generate"  self number_of_series series_length model_parameters seed ))

(defn get-batch-loss 
  "Computes predictions and a loss.

    Args:
      features: A dictionary (such as is produced by a chunker) with the
        following key/value pairs (shapes are given as required for training):
          TrainEvalFeatures.TIMES: A [batch size, self.window_size] integer
            Tensor with times for each observation. To train on longer
            sequences, the data should first be chunked.
          TrainEvalFeatures.VALUES: A [batch size, self.window_size,
            self.num_features] Tensor with values for each observation.
        When evaluating, `TIMES` and `VALUES` must have a window size of at
        least self.window_size, but it may be longer, in which case the last
        window_size - self.input_window_size times (or fewer if this is not
        divisible by self.output_window_size) will be evaluated on with
        non-overlapping output windows (and will have associated
        predictions). This is primarily to support qualitative
        evaluation/plotting, and is not a recommended way to compute evaluation
        losses (since there is no overlap in the output windows, which for
        window-based models is an undesirable bias).
      mode: The tf.estimator.ModeKeys mode to use (TRAIN or EVAL).
      state: Unused
    Returns:
      A model.ModelOutputs object.
    Raises:
      ValueError: If `mode` is not TRAIN or EVAL, or if static shape information
      is incorrect.
    "
  [ self features mode state ]
  (py/call-attr self "get_batch_loss"  self features mode state ))

(defn get-start-state 
  ""
  [ self  ]
  (py/call-attr self "get_start_state"  self  ))

(defn initialize-graph 
  ""
  [ self input_statistics ]
  (py/call-attr self "initialize_graph"  self input_statistics ))

(defn loss-op 
  "Create loss_op."
  [ self targets prediction_ops ]
  (py/call-attr self "loss_op"  self targets prediction_ops ))

(defn predict 
  "Computes predictions multiple steps into the future.

    Args:
      features: A dictionary with the following key/value pairs:
        PredictionFeatures.TIMES: A [batch size, predict window size]
          integer Tensor of times, after the window of data indicated by
          `STATE_TUPLE`, to make predictions for.
        PredictionFeatures.STATE_TUPLE: A tuple of (times, values), times with
          shape [batch size, self.input_window_size], values with shape [batch
          size, self.input_window_size, self.num_features] representing a
          segment of the time series before `TIMES`. This data is used
          to start of the autoregressive computation. This should have data for
          at least self.input_window_size timesteps.
        And any exogenous features, with shapes prefixed by shape of `TIMES`.
    Returns:
      A dictionary with keys, \"mean\", \"covariance\". The
      values are Tensors of shape [batch_size, predict window size,
      num_features] and correspond to the values passed in `TIMES`.
    "
  [ self features ]
  (py/call-attr self "predict"  self features ))

(defn prediction-ops 
  "Compute model predictions given input data.

    Args:
      times: A [batch size, self.window_size] integer Tensor, the first
          self.input_window_size times in each part of the batch indicating
          input features, and the last self.output_window_size times indicating
          prediction times.
      values: A [batch size, self.input_window_size, self.num_features] Tensor
          with input features.
      exogenous_regressors: A [batch size, self.window_size,
          self.exogenous_size] Tensor with exogenous features.
    Returns:
      Tuple (predicted_mean, predicted_covariance), where each element is a
      Tensor with shape [batch size, self.output_window_size,
      self.num_features].
    "
  [ self times values exogenous_regressors ]
  (py/call-attr self "prediction_ops"  self times values exogenous_regressors ))

(defn random-model-parameters 
  ""
  [ self seed ]
  (py/call-attr self "random_model_parameters"  self seed ))
