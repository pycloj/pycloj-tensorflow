(ns tensorflow.contrib.keras.api.keras.datasets.reuters
  "Reuters newswire topic classification dataset."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce reuters (import-module "tensorflow.contrib.keras.api.keras.datasets.reuters"))

(defn get-word-index 
  "Retrieves the dictionary mapping word indices back to words.

  Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).

  Returns:
      The word index dictionary.
  "
  [ & {:keys [path]} ]
   (py/call-attr-kw reuters "get_word_index" [] {:path path }))

(defn load-data 
  "Loads the Reuters newswire classification dataset.

  Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: max number of words to include. Words are ranked
          by how often they occur (in the training set) and only
          the most frequent words are kept
      skip_top: skip the top N most frequently occurring words
          (which may not be informative).
      maxlen: truncate sequences after this length.
      test_split: Fraction of the dataset to be used as test data.
      seed: random seed for sample shuffling.
      start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
      oov_char: words that were cut out because of the `num_words`
          or `skip_top` limit will be replaced with this character.
      index_from: index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Note that the 'out of vocabulary' character is only used for
  words that were present in the training set but are not included
  because they're not making the `num_words` cut here.
  Words that were not seen in the training set but are in the test set
  have simply been skipped.
  "
  [ & {:keys [path num_words skip_top maxlen test_split seed start_char oov_char index_from]
       :or {num_words None maxlen None}} ]
  
   (py/call-attr-kw reuters "load_data" [] {:path path :num_words num_words :skip_top skip_top :maxlen maxlen :test_split test_split :seed seed :start_char start_char :oov_char oov_char :index_from index_from }))
