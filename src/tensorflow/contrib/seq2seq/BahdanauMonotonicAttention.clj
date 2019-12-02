(ns tensorflow.contrib.seq2seq.BahdanauMonotonicAttention
  "Monotonic attention mechanism with Bahadanau-style energy function.

  This type of attention enforces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Since the attention scores are passed
  through a sigmoid, a learnable scalar bias parameter is applied after the
  score function and before the sigmoid.  Otherwise, it is equivalent to
  BahdanauAttention.  This approach is proposed in

  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  \"Online and Linear-Time Attention by Enforcing Monotonic Alignments.\"
  ICML 2017.  https://arxiv.org/abs/1704.00784
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

(defn BahdanauMonotonicAttention 
  "Monotonic attention mechanism with Bahadanau-style energy function.

  This type of attention enforces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Since the attention scores are passed
  through a sigmoid, a learnable scalar bias parameter is applied after the
  score function and before the sigmoid.  Otherwise, it is equivalent to
  BahdanauAttention.  This approach is proposed in

  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  \"Online and Linear-Time Attention by Enforcing Monotonic Alignments.\"
  ICML 2017.  https://arxiv.org/abs/1704.00784
  "
  [num_units memory memory_sequence_length & {:keys [normalize score_mask_value sigmoid_noise sigmoid_noise_seed score_bias_init mode dtype name]
                       :or {score_mask_value None sigmoid_noise_seed None dtype None}} ]
    (py/call-attr-kw seq2seq "BahdanauMonotonicAttention" [num_units memory memory_sequence_length] {:normalize normalize :score_mask_value score_mask_value :sigmoid_noise sigmoid_noise :sigmoid_noise_seed sigmoid_noise_seed :score_bias_init score_bias_init :mode mode :dtype dtype :name name }))

(defn alignments-size 
  ""
  [ self ]
    (py/call-attr self "alignments_size"))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn initial-alignments 
  "Creates the initial alignment values for the monotonic attentions.

    Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length..., 0]
    for all entries in the batch.

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
