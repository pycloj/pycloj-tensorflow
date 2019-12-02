(ns tensorflow.spectral
  "Public API for tf.spectral namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce spectral (import-module "tensorflow.spectral"))

(defn dct 
  "Computes the 1D [Discrete Cosine Transform (DCT)][dct] of `input`.

  Currently only Types I, II and III are supported.
  Type I is implemented using a length `2N` padded `tf.signal.rfft`.
  Type II is implemented using a length `2N` padded `tf.signal.rfft`, as
  described here: [Type 2 DCT using 2N FFT padded (Makhoul)](https://dsp.stackexchange.com/a/10606).
  Type III is a fairly straightforward inverse of Type II
  (i.e. using a length `2N` padded `tf.signal.irfft`).

  @compatibility(scipy)
  Equivalent to [scipy.fftpack.dct](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html)
   for Type-I, Type-II and Type-III DCT.
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to
      take the DCT of.
    type: The DCT type to perform. Must be 1, 2 or 3.
    n: The length of the transform. If length is less than sequence length,
      only the first n elements of the sequence are considered for the DCT.
      If n is greater than the sequence length, zeros are padded and then
      the DCT is computed as usual.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the DCT of `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `axis` is
      not `-1`, `n` is not `None` or greater than 0,
      or `norm` is not `None` or `'ortho'`.
    ValueError: If `type` is `1` and `norm` is `ortho`.

  [dct]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
  "
  [input & {:keys [type n axis norm name]
                       :or {n None norm None name None}} ]
    (py/call-attr-kw spectral "dct" [input] {:type type :n n :axis axis :norm norm :name name }))

(defn fft 
  "Fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform over the inner-most
  dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "fft"  input name ))

(defn fft2d 
  "2D fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform over the inner-most
  2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "fft2d"  input name ))

(defn fft3d 
  "3D fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "fft3d"  input name ))

(defn idct 
  "Computes the 1D [Inverse Discrete Cosine Transform (DCT)][idct] of `input`.

  Currently only Types I, II and III are supported. Type III is the inverse of
  Type II, and vice versa.

  Note that you must re-normalize by 1/(2n) to obtain an inverse if `norm` is
  not `'ortho'`. That is:
  `signal == idct(dct(signal)) * 0.5 / signal.shape[-1]`.
  When `norm='ortho'`, we have:
  `signal == idct(dct(signal, norm='ortho'), norm='ortho')`.

  @compatibility(scipy)
  Equivalent to [scipy.fftpack.idct](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.idct.html)
   for Type-I, Type-II and Type-III DCT.
  @end_compatibility

  Args:
    input: A `[..., samples]` `float32` `Tensor` containing the signals to take
      the DCT of.
    type: The IDCT type to perform. Must be 1, 2 or 3.
    n: For future expansion. The length of the transform. Must be `None`.
    axis: For future expansion. The axis to compute the DCT along. Must be `-1`.
    norm: The normalization to apply. `None` for no normalization or `'ortho'`
      for orthonormal normalization.
    name: An optional name for the operation.

  Returns:
    A `[..., samples]` `float32` `Tensor` containing the IDCT of `input`.

  Raises:
    ValueError: If `type` is not `1`, `2` or `3`, `n` is not `None, `axis` is
      not `-1`, or `norm` is not `None` or `'ortho'`.

  [idct]:
  https://en.wikipedia.org/wiki/Discrete_cosine_transform#Inverse_transforms
  "
  [input & {:keys [type n axis norm name]
                       :or {n None norm None name None}} ]
    (py/call-attr-kw spectral "idct" [input] {:type type :n n :axis axis :norm norm :name name }))

(defn ifft 
  "Inverse fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform over the
  inner-most dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "ifft"  input name ))

(defn ifft2d 
  "Inverse 2D fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform over the
  inner-most 2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "ifft2d"  input name ))

(defn ifft3d 
  "Inverse 3D fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform over the
  inner-most 3 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr spectral "ifft3d"  input name ))

(defn irfft 
  "Inverse real-valued fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most dimension of `input`.

  The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
  `fft_length` is not provided, it is computed from the size of the inner-most
  dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
  compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
  than the corresponding dimension of `input`, the dimension is cropped. If it is
  larger, the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "irfft"  input_tensor fft_length name ))

(defn irfft2d 
  "Inverse 2D real-valued fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 2 dimensions of `input`.

  The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 2 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT2D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "irfft2d"  input_tensor fft_length name ))

(defn irfft3d 
  "Inverse 3D real-valued fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 3 dimensions of `input`.

  The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 3 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT3D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "irfft3d"  input_tensor fft_length name ))

(defn rfft 
  "Real-valued fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most dimension of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
  `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
  followed by the `fft_length / 2` positive-frequency terms.

  Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "rfft"  input_tensor fft_length name ))

(defn rfft2d 
  "2D real-valued fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 2 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "rfft2d"  input_tensor fft_length name ))

(defn rfft3d 
  "3D real-valued fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 3 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  "
  [ input_tensor fft_length name ]
  (py/call-attr spectral "rfft3d"  input_tensor fft_length name ))
