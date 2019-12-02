(ns tensorflow.contrib.seq2seq.BasicDecoder
  "Basic sampling decoder."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn BasicDecoder 
  "Basic sampling decoder."
  [ cell helper initial_state output_layer ]
  (py/call-attr seq2seq "BasicDecoder"  cell helper initial_state output_layer ))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn finalize 
  "Called after decoding iterations complete.

    Args:
      outputs: RNNCell outputs (possibly nested tuple of) tensor[s] for all time
        steps.
      final_state: RNNCell final state (possibly nested tuple of) tensor[s] for
        last time step.
      sequence_lengths: 1-D `int32` tensor containing lengths of each sequence.

    Returns:
      `(final_outputs, final_state)`: `final_outputs` is an object containing
      the final decoder output, `final_state` is a (structure of) state tensors
      and TensorArrays.
    "
  [ self outputs final_state sequence_lengths ]
  (py/call-attr self "finalize"  self outputs final_state sequence_lengths ))

(defn initialize 
  "Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    "
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn output-dtype 
  ""
  [ self ]
    (py/call-attr self "output_dtype"))

(defn output-size 
  ""
  [ self ]
    (py/call-attr self "output_size"))

(defn step 
  "Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    "
  [ self time inputs state name ]
  (py/call-attr self "step"  self time inputs state name ))

(defn tracks-own-finished 
  "Describes whether the Decoder keeps track of finished states.

    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.

    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.

    Returns:
      Python bool.
    "
  [ self ]
    (py/call-attr self "tracks_own_finished"))
