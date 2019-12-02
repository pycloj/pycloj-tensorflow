(ns tensorflow.contrib.seq2seq.BahdanauAttention
  "Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  \"Neural Machine Translation by Jointly Learning to Align and Translate.\"
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  \"Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks.\"
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn BahdanauAttention 
  "Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  \"Neural Machine Translation by Jointly Learning to Align and Translate.\"
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  \"Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks.\"
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.
  "
  [num_units memory memory_sequence_length & {:keys [normalize probability_fn score_mask_value dtype custom_key_value_fn name]
                       :or {probability_fn None score_mask_value None dtype None custom_key_value_fn None}} ]
    (py/call-attr-kw seq2seq "BahdanauAttention" [num_units memory memory_sequence_length] {:normalize normalize :probability_fn probability_fn :score_mask_value score_mask_value :dtype dtype :custom_key_value_fn custom_key_value_fn :name name }))

(defn alignments-size 
  ""
  [ self ]
    (py/call-attr self "alignments_size"))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn initial-alignments 
  "Creates the initial alignment values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return a tensor of all zeros.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    "
  [ self batch_size dtype ]
  (py/call-attr self "initial_alignments"  self batch_size dtype ))

(defn initial-state 
  "Creates the initial state values for the `AttentionWrapper` class.

    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).

    The default behavior is to return the same output as initial_alignments.

    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.

    Returns:
      A structure of all-zero tensors with shapes as described by `state_size`.
    "
  [ self batch_size dtype ]
  (py/call-attr self "initial_state"  self batch_size dtype ))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn memory-layer 
  ""
  [ self ]
    (py/call-attr self "memory_layer"))

(defn query-layer 
  ""
  [ self ]
    (py/call-attr self "query_layer"))

(defn state-size 
  ""
  [ self ]
    (py/call-attr self "state_size"))

(defn values 
  ""
  [ self ]
    (py/call-attr self "values"))
