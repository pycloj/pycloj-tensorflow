(ns tensorflow.contrib.seq2seq.Decoder
  "An RNN Decoder abstract interface object.

  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
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

(defn Decoder 
  "An RNN Decoder abstract interface object.

  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  "
  [  ]
  (py/call-attr seq2seq "Decoder"  ))

(defn batch-size 
  "The batch size of input values."
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
  "Called before any decoding iterations.

    This methods must compute initial input values and initial state.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    "
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn output-dtype 
  "A (possibly nested tuple of...) dtype[s]."
  [ self ]
    (py/call-attr self "output_dtype"))

(defn output-size 
  "A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."
  [ self ]
    (py/call-attr self "output_size"))

(defn step 
  "Called per step of decoding (but only once for dynamic decoding).

    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
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
