(ns tensorflow.contrib.ffmpeg
  "Working with audio using FFmpeg.

See the [FFMPEG](https://tensorflow.org/api_guides/python/contrib.ffmpeg) guide.

@@decode_audio
@@encode_audio
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ffmpeg (import-module "tensorflow.contrib.ffmpeg"))

(defn decode-audio 
  "Create an op that decodes the contents of an audio file. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-09-04.
Instructions for updating:
tf.contrib.ffmpeg will be removed in 2.0, the support for video and audio will continue to be provided in tensorflow-io: https://github.com/tensorflow/io

Note that ffmpeg is free to select the \"best\" audio track from an mp4.
https://trac.ffmpeg.org/wiki/Map

Args:
  contents: The binary contents of the audio file to decode. This is a
      scalar.
  file_format: A string or scalar string tensor specifying which
      format the contents will conform to. This can be mp3, mp4, ogg,
      or wav.
  samples_per_second: The number of samples per second that is
      assumed, as an `int` or scalar `int32` tensor. In some cases,
      resampling will occur to generate the correct sample rate.
  channel_count: The number of channels that should be created from the
      audio contents, as an `int` or scalar `int32` tensor. If the
      `contents` have more than this number, then some channels will
      be merged or dropped. If `contents` has fewer than this, then
      additional channels will be created from the existing ones.
  stream: A string specifying which stream from the content file
      should be decoded, e.g., '0' means the 0-th stream.
      The default value is '' which leaves the decision to ffmpeg.

Returns:
  A rank-2 tensor that has time along dimension 0 and channels along
  dimension 1. Dimension 0 will be `samples_per_second *
  length_in_seconds` wide, and dimension 1 will be `channel_count`
  wide. If ffmpeg fails to decode the audio then an empty tensor will
  be returned."
  [ contents file_format samples_per_second channel_count stream ]
  (py/call-attr ffmpeg "decode_audio"  contents file_format samples_per_second channel_count stream ))

(defn decode-video 
  "Create an op that decodes the contents of a video file. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-09-04.
Instructions for updating:
tf.contrib.ffmpeg will be removed in 2.0, the support for video and audio will continue to be provided in tensorflow-io: https://github.com/tensorflow/io

Args:
  contents: The binary contents of the video file to decode. This is a scalar.

Returns:
  A rank-4 `Tensor` that has `[frames, height, width, 3]` RGB as output."
  [ contents ]
  (py/call-attr ffmpeg "decode_video"  contents ))

(defn encode-audio 
  "Creates an op that encodes an audio file using sampled audio from a tensor. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-09-04.
Instructions for updating:
tf.contrib.ffmpeg will be removed in 2.0, the support for video and audio will continue to be provided in tensorflow-io: https://github.com/tensorflow/io

Args:
  audio: A rank-2 `Tensor` that has time along dimension 0 and
      channels along dimension 1. Dimension 0 is `samples_per_second *
      length_in_seconds` long.
  file_format: The type of file to encode, as a string or rank-0
      string tensor. \"wav\" is the only supported format.
  samples_per_second: The number of samples in the audio tensor per
      second of audio, as an `int` or rank-0 `int32` tensor.

Returns:
  A scalar tensor that contains the encoded audio in the specified file
  format."
  [ audio file_format samples_per_second ]
  (py/call-attr ffmpeg "encode_audio"  audio file_format samples_per_second ))
