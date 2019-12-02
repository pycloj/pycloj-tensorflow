(ns tensorflow.contrib.seq2seq.BeamSearchDecoder
  "BeamSearch sampling decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Meanwhile, with `AttentionWrapper`, coverage penalty is suggested to use
    when computing scores (https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the decoder to cover all inputs.
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
(defn BeamSearchDecoder 
  "BeamSearch sampling decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Meanwhile, with `AttentionWrapper`, coverage penalty is suggested to use
    when computing scores (https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the decoder to cover all inputs.
  "
  [cell embedding start_tokens end_token initial_state beam_width output_layer  & {:keys [length_penalty_weight coverage_penalty_weight reorder_tensor_arrays]} ]
    (py/call-attr-kw seq2seq "BeamSearchDecoder" [cell embedding start_tokens end_token initial_state beam_width output_layer] {:length_penalty_weight length_penalty_weight :coverage_penalty_weight coverage_penalty_weight :reorder_tensor_arrays reorder_tensor_arrays }))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn finalize 
  "Finalize and return the predicted_ids.

    Args:
      outputs: An instance of BeamSearchDecoderOutput.
      final_state: An instance of BeamSearchDecoderState. Passed through to the
        output.
      sequence_lengths: An `int64` tensor shaped `[batch_size, beam_width]`.
        The sequence lengths determined for each beam during decode.
        **NOTE** These are ignored; the updated sequence lengths are stored in
        `final_state.lengths`.

    Returns:
      outputs: An instance of `FinalBeamSearchDecoderOutput` where the
        predicted_ids are the result of calling _gather_tree.
      final_state: The same input instance of `BeamSearchDecoderState`.
    "
  [ self outputs final_state sequence_lengths ]
  (py/call-attr self "finalize"  self outputs final_state sequence_lengths ))

(defn initialize 
  "Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
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
  "The BeamSearchDecoder shuffles its beams and their finished state.

    For this reason, it conflicts with the `dynamic_decode` function's
    tracking of finished states.  Setting this property to true avoids
    early stopping of decoding due to mismanagement of the finished state
    in `dynamic_decode`.

    Returns:
      `True`.
    "
  [ self ]
    (py/call-attr self "tracks_own_finished"))
