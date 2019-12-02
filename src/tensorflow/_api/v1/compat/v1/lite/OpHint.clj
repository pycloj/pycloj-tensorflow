(ns tensorflow.-api.v1.compat.v1.lite.OpHint
  "A class that helps build tflite function invocations.

  It allows you to take a bunch of TensorFlow ops and annotate the construction
  such that toco knows how to convert it to tflite. This embeds a pseudo
  function in a TensorFlow graph. This allows embedding high-level API usage
  information in a lower level TensorFlow implementation so that an alternative
  implementation can be substituted later.

  Essentially, any \"input\" into this pseudo op is fed into an identity, and
  attributes are added to that input before being used by the constituent ops
  that make up the pseudo op. A similar process is done to any output that
  is to be exported from the current op.

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.compat.v1.lite"))

(defn OpHint 
  "A class that helps build tflite function invocations.

  It allows you to take a bunch of TensorFlow ops and annotate the construction
  such that toco knows how to convert it to tflite. This embeds a pseudo
  function in a TensorFlow graph. This allows embedding high-level API usage
  information in a lower level TensorFlow implementation so that an alternative
  implementation can be substituted later.

  Essentially, any \"input\" into this pseudo op is fed into an identity, and
  attributes are added to that input before being used by the constituent ops
  that make up the pseudo op. A similar process is done to any output that
  is to be exported from the current op.

  "
  [function_name & {:keys [level children_inputs_mappings]
                       :or {children_inputs_mappings None}} ]
    (py/call-attr-kw lite "OpHint" [function_name] {:level level :children_inputs_mappings children_inputs_mappings }))

(defn add-input 
  "Add a wrapped input argument to the hint.

    Args:
      *args: The input tensor.
      **kwargs:
        \"name\" label
        \"tag\" a tag to group multiple arguments that will be aggregated. I.e.
          a string like 'cool_input'. Basically multiple inputs can be added
          to the same hint for parallel operations that will eventually be
          combined. An example would be static_rnn which creates multiple copies
          of state or inputs.
        \"aggregate\" aggregation strategy that is valid only for tag non None.
          Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
        \"index_override\" The global index to use. This corresponds to the
          argument order in the final stub that will be generated.
    Returns:
      The wrapped input tensor.
    "
  [ self  ]
  (py/call-attr self "add_input"  self  ))

(defn add-inputs 
  "Add a sequence of inputs to the function invocation.

    Args:
      *args: List of inputs to be converted (should be Tf.Tensor).
      **kwargs: This allows 'names' which should be a list of names.
    Returns:
      Wrapped inputs (identity standins that have additional metadata). These
      are also are also tf.Tensor's.
    "
  [ self  ]
  (py/call-attr self "add_inputs"  self  ))

(defn add-output 
  "Add a wrapped output argument to the hint.

    Args:
      *args: The output tensor.
      **kwargs:
        \"name\" label
        \"tag\" a tag to group multiple arguments that will be aggregated. I.e.
          a string like 'cool_input'. Basically multiple inputs can be added
          to the same hint for parallel operations that will eventually be
          combined. An example would be static_rnn which creates multiple copies
          of state or inputs.
        \"aggregate\" aggregation strategy that is valid only for tag non None.
          Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
        \"index_override\" The global index to use. This corresponds to the
          argument order in the final stub that will be generated.
    Returns:
      The wrapped output tensor.
    "
  [ self  ]
  (py/call-attr self "add_output"  self  ))

(defn add-outputs 
  "Add a sequence of outputs to the function invocation.

    Args:
      *args: List of outputs to be converted (should be tf.Tensor).
      **kwargs: See
    Returns:
      Wrapped outputs (identity standins that have additional metadata). These
      are also tf.Tensor's.
    "
  [ self  ]
  (py/call-attr self "add_outputs"  self  ))
