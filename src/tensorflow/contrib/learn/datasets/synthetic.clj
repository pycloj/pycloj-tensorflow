(ns tensorflow.contrib.learn.python.learn.datasets.synthetic
  "Synthetic dataset generators (deprecated).

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
(defonce synthetic (import-module "tensorflow.contrib.learn.python.learn.datasets.synthetic"))

(defn circles 
  "Create circles separated by some value (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using synthetic datasets from scikits.learn.

Args:
  n_samples: int, number of datapoints to generate
  noise: float or None, standard deviation of the Gaussian noise added
  seed: int or None, seed for the noise
  factor: float, size factor of the inner circles with respect to the outer
    ones
  n_classes: int, number of classes to generate

Returns:
  Shuffled features and labels for 'circles' synthetic dataset of type
  `base.Dataset`

Note:
  The multi-class support might not work as expected if `noise` is enabled

TODO:
  - Generation of unbalanced data

Credit goes to (under BSD 3 clause):
  B. Thirion,
  G. Varoquaux,
  A. Gramfort,
  V. Michel,
  O. Grisel,
  G. Louppe,
  J. Nothman"
  [ & {:keys [n_samples noise seed factor n_classes]
       :or {noise None seed None}} ]
  
   (py/call-attr-kw synthetic "circles" [] {:n_samples n_samples :noise noise :seed seed :factor factor :n_classes n_classes }))
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
    (py/call-attr-kw synthetic "deprecated" [date instructions] {:warn_once warn_once }))

(defn spirals 
  "Create spirals (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using synthetic datasets from scikits.learn.

Currently only binary classification is supported for spiral generation

Args:
  n_samples: int, number of datapoints to generate
  noise: float or None, standard deviation of the Gaussian noise added
  seed: int or None, seed for the noise
  n_loops: int, number of spiral loops, doesn't play well with 'bernoulli'
  mode: str, how the spiral should be generated. Current implementations:
    'archimedes': a spiral with equal distances between branches
    'bernoulli': logarithmic spiral with branch distances increasing
    'fermat': a spiral with branch distances decreasing (sqrt)

Returns:
  Shuffled features and labels for 'spirals' synthetic dataset of type
  `base.Dataset`

Raises:
  ValueError: If the generation `mode` is not valid

TODO:
  - Generation of unbalanced data"
  [ & {:keys [n_samples noise seed mode n_loops]
       :or {noise None seed None}} ]
  
   (py/call-attr-kw synthetic "spirals" [] {:n_samples n_samples :noise noise :seed seed :mode mode :n_loops n_loops }))
