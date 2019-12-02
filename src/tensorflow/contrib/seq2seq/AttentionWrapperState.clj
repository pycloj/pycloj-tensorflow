(ns tensorflow.contrib.seq2seq.AttentionWrapperState
  "`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
    - `attention_state`: A single or tuple of nested objects
       containing attention mechanism state for each attention mechanism.
       The objects may contain Tensors or TensorArrays.
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

(defn AttentionWrapperState 
  "`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
    - `attention_state`: A single or tuple of nested objects
       containing attention mechanism state for each attention mechanism.
       The objects may contain Tensors or TensorArrays.
  "
  [ cell_state attention time alignments alignment_history attention_state ]
  (py/call-attr seq2seq "AttentionWrapperState"  cell_state attention time alignments alignment_history attention_state ))

(defn alignment-history 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "alignment_history"))

(defn alignments 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "alignments"))

(defn attention 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "attention"))

(defn attention-state 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "attention_state"))

(defn cell-state 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "cell_state"))

(defn clone 
  "Clone this object, overriding components provided by kwargs.

    The new state fields' shape must match original state fields' shape. This
    will be validated, and original fields' shape will be propagated to new
    fields.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    "
  [ self  ]
  (py/call-attr self "clone"  self  ))

(defn time 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "time"))
