(ns tensorflow.contrib.learn.python.learn.datasets
  "Dataset utilities and synthetic/reference datasets (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce datasets (import-module "tensorflow.contrib.learn.python.learn.datasets"))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw datasets "deprecated" [date instructions] {:warn_once warn_once }))

(defn load-boston 
  "Load Boston housing dataset. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use scikits.learn.datasets.

Args:
    data_path: string, path to boston dataset (optional)

Returns:
  Dataset object containing data in-memory."
  [ data_path ]
  (py/call-attr datasets "load_boston"  data_path ))
(defn load-dataset 
  "Loads dataset by name. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.data.

Args:
  name: Name of the dataset to load.
  size: Size of the dataset to load.
  test_with_fake_data: If true, load with fake dataset.

Returns:
  Features and labels for given dataset. Can be numpy or iterator.

Raises:
  ValueError: if `name` is not found."
  [name  & {:keys [size test_with_fake_data]} ]
    (py/call-attr-kw datasets "load_dataset" [name] {:size size :test_with_fake_data test_with_fake_data }))

(defn load-iris 
  "Load Iris dataset. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use scikits.learn.datasets.

Args:
    data_path: string, path to iris dataset (optional)

Returns:
  Dataset object containing data in-memory."
  [ data_path ]
  (py/call-attr datasets "load_iris"  data_path ))

(defn make-dataset 
  "Creates binary synthetic datasets. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.data.

Args:
  name: str, name of the dataset to generate
  n_samples: int, number of datapoints to generate
  noise: float or None, standard deviation of the Gaussian noise added
  seed: int or None, seed for noise

Returns:
  Shuffled features and labels for given synthetic dataset of type
  `base.Dataset`

Raises:
  ValueError: Raised if `name` not found

Note:
  - This is a generic synthetic data generator - individual generators might
  have more parameters!
    See documentation for individual parameters
  - Note that the `noise` parameter uses `numpy.random.normal` and depends on
  `numpy`'s seed

TODO:
  - Support multiclass datasets
  - Need shuffling routine. Currently synthetic datasets are reshuffled to
  avoid train/test correlation,
    but that hurts reprodusability"
  [name & {:keys [n_samples noise seed]
                       :or {noise None}} ]
    (py/call-attr-kw datasets "make_dataset" [name] {:n_samples n_samples :noise noise :seed seed }))
