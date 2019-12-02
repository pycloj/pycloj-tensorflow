(ns tensorflow.python.keras.api.-v1.keras.preprocessing.text
  "Utilities for text input preprocessing.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce text (import-module "tensorflow.python.keras.api._v1.keras.preprocessing.text"))
(defn hashing-trick 
  "Converts a text to a sequence of indexes in a fixed-size hashing space.

    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: ``!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.

    # Returns
        A list of integer word indices (unicity non-guaranteed).

    `0` is a reserved index that won't be assigned to any word.

    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    "
  [text n hash_function  & {:keys [filters lower split]} ]
    (py/call-attr-kw text "hashing_trick" [text n hash_function] {:filters filters :lower lower :split split }))
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
