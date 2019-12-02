(ns tensorflow.python.keras.api.-v1.keras.preprocessing.sequence.TimeseriesGenerator
  "Utility class for generating batches of temporal data.
  This class takes in a sequence of data-points gathered at
  equal intervals, along with time series parameters such as
  stride, length of history, etc., to produce batches for
  training/validation.
  # Arguments
      data: Indexable generator (such as list or Numpy array)
          containing consecutive data points (timesteps).
          The data should be at 2D, and axis 0 is expected
          to be the time dimension.
      targets: Targets corresponding to timesteps in `data`.
          It should have same length as `data`.
      length: Length of the output sequences (in number of timesteps).
      sampling_rate: Period between successive individual timesteps
          within sequences. For rate `r`, timesteps
          `data[i]`, `data[i-r]`, ... `data[i - length]`
          are used for create a sample sequence.
      stride: Period between successive output sequences.
          For stride `s`, consecutive output samples would
          be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
      start_index: Data points earlier than `start_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      end_index: Data points later than `end_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      shuffle: Whether to shuffle output samples,
          or instead draw them in chronological order.
      reverse: Boolean: if `true`, timesteps in each output sample will be
          in reverse chronological order.
      batch_size: Number of timeseries samples in each batch
          (except maybe the last one).
  # Returns
      A [Sequence](/utils/#sequence) instance.
  # Examples
  ```python
  from keras.preprocessing.sequence import TimeseriesGenerator
  import numpy as np
  data = np.array([[i] for i in range(50)])
  targets = np.array([[i] for i in range(50)])
  data_gen = TimeseriesGenerator(data, targets,
                                 length=10, sampling_rate=2,
                                 batch_size=2)
  assert len(data_gen) == 20
  batch_0 = data_gen[0]
  x, y = batch_0
  assert np.array_equal(x,
                        np.array([[[0], [2], [4], [6], [8]],
                                  [[1], [3], [5], [7], [9]]]))
  assert np.array_equal(y,
                        np.array([[10], [11]]))
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sequence (import-module "tensorflow.python.keras.api._v1.keras.preprocessing.sequence"))

(defn TimeseriesGenerator 
  "Utility class for generating batches of temporal data.
  This class takes in a sequence of data-points gathered at
  equal intervals, along with time series parameters such as
  stride, length of history, etc., to produce batches for
  training/validation.
  # Arguments
      data: Indexable generator (such as list or Numpy array)
          containing consecutive data points (timesteps).
          The data should be at 2D, and axis 0 is expected
          to be the time dimension.
      targets: Targets corresponding to timesteps in `data`.
          It should have same length as `data`.
      length: Length of the output sequences (in number of timesteps).
      sampling_rate: Period between successive individual timesteps
          within sequences. For rate `r`, timesteps
          `data[i]`, `data[i-r]`, ... `data[i - length]`
          are used for create a sample sequence.
      stride: Period between successive output sequences.
          For stride `s`, consecutive output samples would
          be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
      start_index: Data points earlier than `start_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      end_index: Data points later than `end_index` will not be used
          in the output sequences. This is useful to reserve part of the
          data for test or validation.
      shuffle: Whether to shuffle output samples,
          or instead draw them in chronological order.
      reverse: Boolean: if `true`, timesteps in each output sample will be
          in reverse chronological order.
      batch_size: Number of timeseries samples in each batch
          (except maybe the last one).
  # Returns
      A [Sequence](/utils/#sequence) instance.
  # Examples
  ```python
  from keras.preprocessing.sequence import TimeseriesGenerator
  import numpy as np
  data = np.array([[i] for i in range(50)])
  targets = np.array([[i] for i in range(50)])
  data_gen = TimeseriesGenerator(data, targets,
                                 length=10, sampling_rate=2,
                                 batch_size=2)
  assert len(data_gen) == 20
  batch_0 = data_gen[0]
  x, y = batch_0
  assert np.array_equal(x,
                        np.array([[[0], [2], [4], [6], [8]],
                                  [[1], [3], [5], [7], [9]]]))
  assert np.array_equal(y,
                        np.array([[10], [11]]))
  ```
  "
  [data targets length & {:keys [sampling_rate stride start_index end_index shuffle reverse batch_size]
                       :or {end_index None}} ]
    (py/call-attr-kw sequence "TimeseriesGenerator" [data targets length] {:sampling_rate sampling_rate :stride stride :start_index start_index :end_index end_index :shuffle shuffle :reverse reverse :batch_size batch_size }))

(defn get-config 
  "Returns the TimeseriesGenerator configuration as Python dictionary.

        # Returns
            A Python dictionary with the TimeseriesGenerator configuration.
        "
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn on-epoch-end 
  "Method called at the end of every epoch.
    "
  [ self  ]
  (py/call-attr self "on_epoch_end"  self  ))

(defn to-json 
  "Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        "
  [ self  ]
  (py/call-attr self "to_json"  self  ))
