(ns tensorflow-core.contrib.signal
  "Signal processing operations.

`tf.contrib.signal` has been renamed to `tf.signal`. `tf.contrib.signal` will be
removed in TensorFlow 2.0.

See the
[Contrib Signal](https://tensorflow.org/api_guides/python/contrib.signal)
guide.

@@frame
@@hamming_window
@@hann_window
@@inverse_stft
@@inverse_stft_window_fn
@@mfccs_from_log_mel_spectrograms
@@linear_to_mel_weight_matrix
@@overlap_and_add
@@stft

[hamming]: https://en.wikipedia.org/wiki/Window_function#Hamming_window
[hann]: https://en.wikipedia.org/wiki/Window_function#Hann_window
[mel]: https://en.wikipedia.org/wiki/Mel_scale
[mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
[stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce signal (import-module "tensorflow_core.contrib.signal"))

(defn frame 
  "Expands `signal`'s `axis` dimension into frames of `frame_length`.

  Slides a window of size `frame_length` over `signal`'s `axis` dimension
  with a stride of `frame_step`, replacing the `axis` dimension with
  `[frames, frame_length]` frames.

  If `pad_end` is True, window positions that are past the end of the `axis`
  dimension are padded with `pad_value` until the window moves fully past the
  end of the dimension. Otherwise, only window positions that fully overlap the
  `axis` dimension are produced.

  For example:

  ```python
  pcm = tf.compat.v1.placeholder(tf.float32, [None, 9152])
  frames = tf.signal.frame(pcm, 512, 180)
  magspec = tf.abs(tf.signal.rfft(frames, [512]))
  image = tf.expand_dims(magspec, 3)
  ```

  Args:
    signal: A `[..., samples, ...]` `Tensor`. The rank and dimensions
      may be unknown. Rank must be at least 1.
    frame_length: The frame length in samples. An integer or scalar `Tensor`.
    frame_step: The frame hop size in samples. An integer or scalar `Tensor`.
    pad_end: Whether to pad the end of `signal` with `pad_value`.
    pad_value: An optional scalar `Tensor` to use where the input signal
      does not exist when `pad_end` is True.
    axis: A scalar integer `Tensor` indicating the axis to frame. Defaults to
      the last axis. Supports negative values for indexing from the end.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of frames with shape `[..., frames, frame_length, ...]`.

  Raises:
    ValueError: If `frame_length`, `frame_step`, `pad_value`, or `axis` are not
      scalar.
  "
  [signal frame_length frame_step & {:keys [pad_end pad_value axis name]
                       :or {name None}} ]
    (py/call-attr-kw signal "frame" [signal frame_length frame_step] {:pad_end pad_end :pad_value pad_value :axis axis :name name }))

(defn hamming-window 
  "Generate a [Hamming][hamming] window.

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window. Periodic windows are typically used for spectral
      analysis while symmetric windows are typically used for digital
      filter design.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type.

  [hamming]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
  "
  [window_length & {:keys [periodic dtype name]
                       :or {name None}} ]
    (py/call-attr-kw signal "hamming_window" [window_length] {:periodic periodic :dtype dtype :name name }))

(defn hann-window 
  "Generate a [Hann window][hann].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    periodic: A bool `Tensor` indicating whether to generate a periodic or
      symmetric window. Periodic windows are typically used for spectral
      analysis while symmetric windows are typically used for digital
      filter design.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  Raises:
    ValueError: If `dtype` is not a floating point type.

  [hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
  "
  [window_length & {:keys [periodic dtype name]
                       :or {name None}} ]
    (py/call-attr-kw signal "hann_window" [window_length] {:periodic periodic :dtype dtype :name name }))

(defn inverse-stft 
  "Computes the inverse [Short-time Fourier Transform][stft] of `stfts`.

  To reconstruct an original waveform, a complimentary window function should
  be used in inverse_stft. Such a window function can be constructed with
  tf.signal.inverse_stft_window_fn.

  Example:

  ```python
  frame_length = 400
  frame_step = 160
  waveform = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(waveform, frame_length, frame_step)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  ```

  if a custom window_fn is used in stft, it must be passed to
  inverse_stft_window_fn:

  ```python
  frame_length = 400
  frame_step = 160
  window_fn = functools.partial(window_ops.hamming_window, periodic=True),
  waveform = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(
      waveform, frame_length, frame_step, window_fn=window_fn)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step, forward_window_fn=window_fn))
  ```

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    stfts: A `complex64` `[..., frames, fft_unique_bins]` `Tensor` of STFT bins
      representing a batch of `fft_length`-point STFTs where `fft_unique_bins`
      is `fft_length // 2 + 1`
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT that produced
      `stfts`. If not provided, uses the smallest power of 2 enclosing
      `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `Tensor` of `float32` signals representing the inverse
    STFT for each input STFT in `stfts`.

  Raises:
    ValueError: If `stfts` is not at least rank 2, `frame_length` is not scalar,
      `frame_step` is not scalar, or `fft_length` is not scalar.

  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  "
  [stfts frame_length frame_step fft_length & {:keys [window_fn name]
                       :or {name None}} ]
    (py/call-attr-kw signal "inverse_stft" [stfts frame_length frame_step fft_length] {:window_fn window_fn :name name }))

(defn inverse-stft-window-fn 
  "Generates a window function that can be used in `inverse_stft`.

  Constructs a window that is equal to the forward window with a further
  pointwise amplitude correction.  `inverse_stft_window_fn` is equivalent to
  `forward_window_fn` in the case where it would produce an exact inverse.

  See examples in `inverse_stft` documentation for usage.

  Args:
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    forward_window_fn: window_fn used in the forward transform, `stft`.
    name: An optional name for the operation.

  Returns:
    A callable that takes a window length and a `dtype` keyword argument and
      returns a `[window_length]` `Tensor` of samples in the provided datatype.
      The returned window is suitable for reconstructing original waveform in
      inverse_stft.
  "
  [frame_step & {:keys [forward_window_fn name]
                       :or {name None}} ]
    (py/call-attr-kw signal "inverse_stft_window_fn" [frame_step] {:forward_window_fn forward_window_fn :name name }))

(defn linear-to-mel-weight-matrix 
  "Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].

  Returns a weight matrix that can be used to re-weight a `Tensor` containing
  `num_spectrogram_bins` linearly sampled frequency information from
  `[0, sample_rate / 2]` into `num_mel_bins` frequency information from
  `[lower_edge_hertz, upper_edge_hertz]` on the [mel scale][mel].

  For example, the returned matrix `A` can be used to right-multiply a
  spectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear
  scale spectrum values (e.g. STFT magnitudes) to generate a \"mel spectrogram\"
  `M` of shape `[frames, num_mel_bins]`.

      # `S` has shape [frames, num_spectrogram_bins]
      # `M` has shape [frames, num_mel_bins]
      M = tf.matmul(S, A)

  The matrix can be used with `tf.tensordot` to convert an arbitrary rank
  `Tensor` of linear-scale spectral bins into the mel scale.

      # S has shape [..., num_spectrogram_bins].
      # M has shape [..., num_mel_bins].
      M = tf.tensordot(S, A, 1)
      # tf.tensordot does not support shape inference for this case yet.
      M.set_shape(S.shape[:-1].concatenate(A.shape[-1:]))

  Args:
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
      source spectrogram data, which is understood to be `fft_size // 2 + 1`,
      i.e. the spectrogram only contains the nonredundant FFT bins.
    sample_rate: Python float. Samples per second of the input signal used to
      create the spectrogram. We need this to figure out the actual frequencies
      for each spectrogram bin, which dictates how they are mapped into the mel
      scale.
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The `DType` of the result matrix. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[num_spectrogram_bins, num_mel_bins]`.

  Raises:
    ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
      positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
      ordered, `upper_edge_hertz` is larger than the Nyquist frequency, or
      `sample_rate` is neither a Python float nor a constant Tensor.

  [mel]: https://en.wikipedia.org/wiki/Mel_scale
  "
  [ & {:keys [num_mel_bins num_spectrogram_bins sample_rate lower_edge_hertz upper_edge_hertz dtype name]
       :or {name None}} ]
  
   (py/call-attr-kw signal "linear_to_mel_weight_matrix" [] {:num_mel_bins num_mel_bins :num_spectrogram_bins num_spectrogram_bins :sample_rate sample_rate :lower_edge_hertz lower_edge_hertz :upper_edge_hertz upper_edge_hertz :dtype dtype :name name }))

(defn mfccs-from-log-mel-spectrograms 
  "Computes [MFCCs][mfcc] of `log_mel_spectrograms`.

  Implemented with GPU-compatible ops and supports gradients.

  [Mel-Frequency Cepstral Coefficient (MFCC)][mfcc] calculation consists of
  taking the DCT-II of a log-magnitude mel-scale spectrogram. [HTK][htk]'s MFCCs
  use a particular scaling of the DCT-II which is almost orthogonal
  normalization. We follow this convention.

  All `num_mel_bins` MFCCs are returned and it is up to the caller to select
  a subset of the MFCCs based on their application. For example, it is typical
  to only use the first few for speech recognition, as this results in
  an approximately pitch-invariant representation of the signal.

  For example:

  ```python
  sample_rate = 16000.0
  # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
  pcm = tf.compat.v1.placeholder(tf.float32, [None, None])

  # A 1024-point STFT with frames of 64 ms and 75% overlap.
  stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256,
                         fft_length=1024)
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1].value
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]
  ```

  Args:
    log_mel_spectrograms: A `[..., num_mel_bins]` `float32` `Tensor` of
      log-magnitude mel-scale spectrograms.
    name: An optional name for the operation.
  Returns:
    A `[..., num_mel_bins]` `float32` `Tensor` of the MFCCs of
    `log_mel_spectrograms`.

  Raises:
    ValueError: If `num_mel_bins` is not positive.

  [mfcc]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
  [htk]: https://en.wikipedia.org/wiki/HTK_(software)
  "
  [ log_mel_spectrograms name ]
  (py/call-attr signal "mfccs_from_log_mel_spectrograms"  log_mel_spectrograms name ))

(defn overlap-and-add 
  "Reconstructs a signal from a framed representation.

  Adds potentially overlapping frames of a signal with shape
  `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
  The resulting tensor has shape `[..., output_size]` where

      output_size = (frames - 1) * frame_step + frame_length

  Args:
    signal: A [..., frames, frame_length] `Tensor`. All dimensions may be
      unknown, and rank must be at least 2.
    frame_step: An integer or scalar `Tensor` denoting overlap offsets. Must be
      less than or equal to `frame_length`.
    name: An optional name for the operation.

  Returns:
    A `Tensor` with shape `[..., output_size]` containing the overlap-added
    frames of `signal`'s inner-most two dimensions.

  Raises:
    ValueError: If `signal`'s rank is less than 2, or `frame_step` is not a
      scalar integer.
  "
  [ signal frame_step name ]
  (py/call-attr signal "overlap_and_add"  signal frame_step name ))

(defn stft 
  "Computes the [Short-time Fourier Transform][stft] of `signals`.

  Implemented with GPU-compatible ops and supports gradients.

  Args:
    signals: A `[..., samples]` `float32` `Tensor` of real-valued signals.
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT to apply.
      If not provided, uses the smallest power of 2 enclosing `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    name: An optional name for the operation.

  Returns:
    A `[..., frames, fft_unique_bins]` `Tensor` of `complex64` STFT values where
    `fft_unique_bins` is `fft_length // 2 + 1` (the unique components of the
    FFT).

  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_step` is not scalar.

  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  "
  [signals frame_length frame_step fft_length & {:keys [window_fn pad_end name]
                       :or {name None}} ]
    (py/call-attr-kw signal "stft" [signals frame_length frame_step fft_length] {:window_fn window_fn :pad_end pad_end :name name }))
