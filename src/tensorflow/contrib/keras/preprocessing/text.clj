(ns tensorflow.contrib.keras.api.keras.preprocessing.text
  "Keras data preprocessing utils for text data."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce text (import-module "tensorflow.contrib.keras.api.keras.preprocessing.text"))
(defn one-hot 
  "One-hot encodes a text into a list of word indexes of size n.

    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.

    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.

    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    "
  [text n  & {:keys [filters lower split]} ]
    (py/call-attr-kw text "one_hot" [text n] {:filters filters :lower lower :split split }))
(defn text-to-word-sequence 
  "Converts a text to a sequence of words (or tokens).

    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    # Returns
        A list of words (or tokens).
    "
  [text  & {:keys [filters lower split]} ]
    (py/call-attr-kw text "text_to_word_sequence" [text] {:filters filters :lower lower :split split }))
