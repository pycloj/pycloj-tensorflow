(ns tensorflow.contrib.learn.python.learn.preprocessing.categorical.CategoricalProcessor
  "Maps documents to sequences of word ids.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  As a common convention, Nan values are handled as unknown tokens.
  Both float('nan') and np.nan are accepted.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce categorical (import-module "tensorflow.contrib.learn.python.learn.preprocessing.categorical"))

(defn CategoricalProcessor 
  "Maps documents to sequences of word ids.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  As a common convention, Nan values are handled as unknown tokens.
  Both float('nan') and np.nan are accepted.
  "
  [ & {:keys [min_frequency share vocabularies]
       :or {vocabularies None}} ]
  
   (py/call-attr-kw categorical "CategoricalProcessor" [] {:min_frequency min_frequency :share share :vocabularies vocabularies }))

(defn fit 
  "Learn a vocabulary dictionary of all categories in `x`.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.
      unused_y: to match fit format signature of estimators.

    Returns:
      self
    "
  [ self x unused_y ]
  (py/call-attr self "fit"  self x unused_y ))

(defn fit-transform 
  "Learn the vocabulary dictionary and return indexies of categories.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.
      unused_y: to match fit_transform signature of estimators.

    Returns:
      x: iterable, [n_samples]. Category-id matrix.
    "
  [ self x unused_y ]
  (py/call-attr self "fit_transform"  self x unused_y ))
(defn freeze 
  "Freeze or unfreeze all vocabularies.

    Args:
      freeze: Boolean, indicate if vocabularies should be frozen.
    "
  [self   & {:keys [freeze]} ]
    (py/call-attr-kw self "freeze" [] {:freeze freeze }))

(defn transform 
  "Transform documents to category-id matrix.

    Converts categories to ids give fitted vocabulary from `fit` or
    one provided in the constructor.

    Args:
      x: numpy matrix or iterable of lists/numpy arrays.

    Yields:
      x: iterable, [n_samples]. Category-id matrix.
    "
  [ self x ]
  (py/call-attr self "transform"  self x ))
