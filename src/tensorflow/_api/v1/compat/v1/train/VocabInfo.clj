(ns tensorflow.-api.v1.compat.v1.train.VocabInfo
  "Vocabulary information for warm-starting.

  See `tf.estimator.WarmStartSettings` for examples of using
  VocabInfo to warm-start.

  Args:
    new_vocab: [Required] A path to the new vocabulary file (used with the model
      to be trained).
    new_vocab_size: [Required] An integer indicating how many entries of the new
      vocabulary will used in training.
    num_oov_buckets: [Required] An integer indicating how many OOV buckets are
      associated with the vocabulary.
    old_vocab: [Required] A path to the old vocabulary file (used with the
      checkpoint to be warm-started from).
    old_vocab_size: [Optional] An integer indicating how many entries of the old
      vocabulary were used in the creation of the checkpoint. If not provided,
      the entire old vocabulary will be used.
    backup_initializer: [Optional] A variable initializer used for variables
      corresponding to new vocabulary entries and OOV. If not provided, these
      entries will be zero-initialized.
    axis: [Optional] Denotes what axis the vocabulary corresponds to.  The
      default, 0, corresponds to the most common use case (embeddings or
      linear weights for binary classification / regression).  An axis of 1
      could be used for warm-starting output layers with class vocabularies.

  Returns:
    A `VocabInfo` which represents the vocabulary information for warm-starting.

  Raises:
    ValueError: `axis` is neither 0 or 1.

      Example Usage:
```python
      embeddings_vocab_info = tf.VocabInfo(
          new_vocab='embeddings_vocab',
          new_vocab_size=100,
          num_oov_buckets=1,
          old_vocab='pretrained_embeddings_vocab',
          old_vocab_size=10000,
          backup_initializer=tf.compat.v1.truncated_normal_initializer(
              mean=0.0, stddev=(1 / math.sqrt(embedding_dim))),
          axis=0)

      softmax_output_layer_kernel_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.glorot_uniform_initializer(),
          axis=1)

      softmax_output_layer_bias_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.zeros_initializer(),
          axis=0)

      #Currently, only axis=0 and axis=1 are supported.
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
(defonce train (import-module "tensorflow._api.v1.compat.v1.train"))

(defn VocabInfo 
  "Vocabulary information for warm-starting.

  See `tf.estimator.WarmStartSettings` for examples of using
  VocabInfo to warm-start.

  Args:
    new_vocab: [Required] A path to the new vocabulary file (used with the model
      to be trained).
    new_vocab_size: [Required] An integer indicating how many entries of the new
      vocabulary will used in training.
    num_oov_buckets: [Required] An integer indicating how many OOV buckets are
      associated with the vocabulary.
    old_vocab: [Required] A path to the old vocabulary file (used with the
      checkpoint to be warm-started from).
    old_vocab_size: [Optional] An integer indicating how many entries of the old
      vocabulary were used in the creation of the checkpoint. If not provided,
      the entire old vocabulary will be used.
    backup_initializer: [Optional] A variable initializer used for variables
      corresponding to new vocabulary entries and OOV. If not provided, these
      entries will be zero-initialized.
    axis: [Optional] Denotes what axis the vocabulary corresponds to.  The
      default, 0, corresponds to the most common use case (embeddings or
      linear weights for binary classification / regression).  An axis of 1
      could be used for warm-starting output layers with class vocabularies.

  Returns:
    A `VocabInfo` which represents the vocabulary information for warm-starting.

  Raises:
    ValueError: `axis` is neither 0 or 1.

      Example Usage:
```python
      embeddings_vocab_info = tf.VocabInfo(
          new_vocab='embeddings_vocab',
          new_vocab_size=100,
          num_oov_buckets=1,
          old_vocab='pretrained_embeddings_vocab',
          old_vocab_size=10000,
          backup_initializer=tf.compat.v1.truncated_normal_initializer(
              mean=0.0, stddev=(1 / math.sqrt(embedding_dim))),
          axis=0)

      softmax_output_layer_kernel_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.glorot_uniform_initializer(),
          axis=1)

      softmax_output_layer_bias_vocab_info = tf.VocabInfo(
          new_vocab='class_vocab',
          new_vocab_size=5,
          num_oov_buckets=0,  # No OOV for classes.
          old_vocab='old_class_vocab',
          old_vocab_size=8,
          backup_initializer=tf.compat.v1.zeros_initializer(),
          axis=0)

      #Currently, only axis=0 and axis=1 are supported.
  ```
  "
  [new_vocab new_vocab_size num_oov_buckets old_vocab & {:keys [old_vocab_size backup_initializer axis]
                       :or {backup_initializer None}} ]
    (py/call-attr-kw train "VocabInfo" [new_vocab new_vocab_size num_oov_buckets old_vocab] {:old_vocab_size old_vocab_size :backup_initializer backup_initializer :axis axis }))

(defn axis 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "axis"))

(defn backup-initializer 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "backup_initializer"))

(defn new-vocab 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "new_vocab"))

(defn new-vocab-size 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "new_vocab_size"))

(defn num-oov-buckets 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "num_oov_buckets"))

(defn old-vocab 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "old_vocab"))

(defn old-vocab-size 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "old_vocab_size"))
