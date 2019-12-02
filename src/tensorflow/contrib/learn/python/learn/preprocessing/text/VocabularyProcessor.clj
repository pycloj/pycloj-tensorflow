(ns tensorflow.contrib.learn.python.learn.preprocessing.text.VocabularyProcessor
  "Maps documents to sequences of word ids.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce text (import-module "tensorflow.contrib.learn.python.learn.preprocessing.text"))

(defn VocabularyProcessor 
  "Maps documents to sequences of word ids.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [max_document_length & {:keys [min_frequency vocabulary tokenizer_fn]
                       :or {vocabulary None tokenizer_fn None}} ]
    (py/call-attr-kw text "VocabularyProcessor" [max_document_length] {:min_frequency min_frequency :vocabulary vocabulary :tokenizer_fn tokenizer_fn }))

(defn fit 
  "Learn a vocabulary dictionary of all tokens in the raw documents.

    Args:
      raw_documents: An iterable which yield either str or unicode.
      unused_y: to match fit format signature of estimators.

    Returns:
      self
    "
  [ self raw_documents unused_y ]
  (py/call-attr self "fit"  self raw_documents unused_y ))

(defn fit-transform 
  "Learn the vocabulary dictionary and return indexies of words.

    Args:
      raw_documents: An iterable which yield either str or unicode.
      unused_y: to match fit_transform signature of estimators.

    Returns:
      x: iterable, [n_samples, max_document_length]. Word-id matrix.
    "
  [ self raw_documents unused_y ]
  (py/call-attr self "fit_transform"  self raw_documents unused_y ))

(defn reverse 
  "Reverses output of vocabulary mapping to words.

    Args:
      documents: iterable, list of class ids.

    Yields:
      Iterator over mapped in words documents.
    "
  [ self documents ]
  (py/call-attr self "reverse"  self documents ))

(defn save 
  "Saves vocabulary processor into given file.

    Args:
      filename: Path to output file.
    "
  [ self filename ]
  (py/call-attr self "save"  self filename ))

(defn transform 
  "Transform documents to word-id matrix.

    Convert words to ids with vocabulary fitted with fit or the one
    provided in the constructor.

    Args:
      raw_documents: An iterable which yield either str or unicode.

    Yields:
      x: iterable, [n_samples, max_document_length]. Word-id matrix.
    "
  [ self raw_documents ]
  (py/call-attr self "transform"  self raw_documents ))
