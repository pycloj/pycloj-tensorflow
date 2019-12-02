(ns tensorflow.contrib.seq2seq.FinalBeamSearchDecoderOutput
  "Final outputs returned by the beam search after all decoding is finished.

  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if
      `output_time_major` is True). Beams are ordered from best to worst.
    beam_search_decoder_output: An instance of `BeamSearchDecoderOutput` that
      describes the state of the beam search.
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

(defn FinalBeamSearchDecoderOutput 
  "Final outputs returned by the beam search after all decoding is finished.

  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if
      `output_time_major` is True). Beams are ordered from best to worst.
    beam_search_decoder_output: An instance of `BeamSearchDecoderOutput` that
      describes the state of the beam search.
  "
  [ predicted_ids beam_search_decoder_output ]
  (py/call-attr seq2seq "FinalBeamSearchDecoderOutput"  predicted_ids beam_search_decoder_output ))

(defn beam-search-decoder-output 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "beam_search_decoder_output"))

(defn predicted-ids 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "predicted_ids"))
