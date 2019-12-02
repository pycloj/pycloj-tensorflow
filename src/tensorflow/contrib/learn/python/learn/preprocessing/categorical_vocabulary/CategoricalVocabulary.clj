(ns tensorflow.contrib.learn.python.learn.preprocessing.categorical-vocabulary.CategoricalVocabulary
  "Categorical variables vocabulary class.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Accumulates and provides mapping from classes to indexes.
  Can be easily used for words.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce categorical-vocabulary (import-module "tensorflow.contrib.learn.python.learn.preprocessing.categorical_vocabulary"))

(defn CategoricalVocabulary 
  "Categorical variables vocabulary class.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Accumulates and provides mapping from classes to indexes.
  Can be easily used for words.
  "
  [ & {:keys [unknown_token support_reverse]} ]
   (py/call-attr-kw categorical-vocabulary "CategoricalVocabulary" [] {:unknown_token unknown_token :support_reverse support_reverse }))
(defn add 
  "Adds count of the category to the frequency table.

    Args:
      category: string or integer, category to add frequency to.
      count: optional integer, how many to add.
    "
  [self category  & {:keys [count]} ]
    (py/call-attr-kw self "add" [category] {:count count }))
(defn freeze 
  "Freezes the vocabulary, after which new words return unknown token id.

    Args:
      freeze: True to freeze, False to unfreeze.
    "
  [self   & {:keys [freeze]} ]
    (py/call-attr-kw self "freeze" [] {:freeze freeze }))

(defn get 
  "Returns word's id in the vocabulary.

    If category is new, creates a new id for it.

    Args:
      category: string or integer to lookup in vocabulary.

    Returns:
      interger, id in the vocabulary.
    "
  [ self category ]
  (py/call-attr self "get"  self category ))

(defn reverse 
  "Given class id reverse to original class name.

    Args:
      class_id: Id of the class.

    Returns:
      Class name.

    Raises:
      ValueError: if this vocabulary wasn't initialized with support_reverse.
    "
  [ self class_id ]
  (py/call-attr self "reverse"  self class_id ))
(defn trim 
  "Trims vocabulary for minimum frequency.

    Remaps ids from 1..n in sort frequency order.
    where n - number of elements left.

    Args:
      min_frequency: minimum frequency to keep.
      max_frequency: optional, maximum frequency to keep.
        Useful to remove very frequent categories (like stop words).
    "
  [self min_frequency  & {:keys [max_frequency]} ]
    (py/call-attr-kw self "trim" [min_frequency] {:max_frequency max_frequency }))
