(ns tensorflow.-api.v1.compat.v1.quantization
  "Public API for tf.quantization namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce quantization (import-module "tensorflow._api.v1.compat.v1.quantization"))

(defn dequantize 
  "Dequantize the 'input' tensor into a float Tensor.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  if T == qint8: in[i] += (range(T) + 1)/ 2.0
  out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
  ```
  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  If the input comes from a QuantizedRelu6, the output type is
  quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
  0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
  Dequantize on quint8 will take each value, cast to float, and multiply
  by 6 / 255.
  Note that if quantizedtype is qint8, the operation will additionally add
  each value by 128 prior to casting.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```c++
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = range / num_discrete_values
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
  ```

  *SCALED mode Example*

  `SCALED` mode matches the quantization approach used in
  `QuantizeAndDequantize{V2|V3}`.

  If the mode is `SCALED`, we do not use the full range of the output type,
  choosing to elide the lowest possible value for symmetry (e.g., output range is
  -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
  0.

  We first find the range of values in our tensor. The
  range we use is always centered on 0, so we find m such that
  ```c++
    m = max(abs(input_min), abs(input_max))
  ```

  Our input tensor range is then `[-m, m]`.

  Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
  If T is signed, this is
  ```
    num_bits = sizeof(T) * 8
    [min_fixed, max_fixed] =
        [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
  ```

  Otherwise, if T is unsigned, the fixed-point range is
  ```
    [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
  ```

  From this we compute our scaling factor, s:
  ```c++
    s = (2 * m) / (max_fixed - min_fixed)
  ```

  Now we can dequantize the elements of our tensor:
  ```c++
  result = input * s
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    mode: An optional `string` from: `\"MIN_COMBINED\", \"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_COMBINED\"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [input min_range max_range & {:keys [mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "dequantize" [input min_range max_range] {:mode mode :name name }))

(defn fake-quant-with-min-max-args 
  "Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

  Attributes `[min; max]` define the clamping range for the `inputs` data.
  `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
  when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
  then de-quantized and output as floats in `[min; max]` interval.
  `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:
  If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  Quantization is called fake since the output is still in floating point.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: An optional `float`. Defaults to `-6`.
    max: An optional `float`. Defaults to `6`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [inputs & {:keys [min max num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_args" [inputs] {:min min :max max :num_bits num_bits :narrow_range narrow_range :name name }))

(defn fake-quant-with-min-max-args-gradient 
  "Compute gradients for a FakeQuantWithMinMaxArgs operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
    min: An optional `float`. Defaults to `-6`.
    max: An optional `float`. Defaults to `6`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [gradients inputs & {:keys [min max num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_args_gradient" [gradients inputs] {:min min :max max :num_bits num_bits :narrow_range narrow_range :name name }))

(defn fake-quant-with-min-max-vars 
  "Fake-quantize the 'inputs' tensor of type float via global float scalars `min`

  and `max` to 'outputs' tensor of same shape as `inputs`.

  `[min; max]` define the clamping range for the `inputs` data.
  `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
  when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
  then de-quantized and output as floats in `[min; max]` interval.
  `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:
  If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  This operation has a gradient and thus allows for training `min` and `max`
  values.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [inputs min max & {:keys [num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_vars" [inputs min max] {:num_bits num_bits :narrow_range narrow_range :name name }))

(defn fake-quant-with-min-max-vars-gradient 
  "Compute gradients for a FakeQuantWithMinMaxVars operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxVars operation.
      min, max: Quantization interval, scalar floats.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization; between 2 and 8, inclusive.
    narrow_range: An optional `bool`. Defaults to `False`.
      Whether to quantize into 2^num_bits - 1 distinct values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

    backprops_wrt_input: A `Tensor` of type `float32`.
    backprop_wrt_min: A `Tensor` of type `float32`.
    backprop_wrt_max: A `Tensor` of type `float32`.
  "
  [gradients inputs min max & {:keys [num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_vars_gradient" [gradients inputs min max] {:num_bits num_bits :narrow_range narrow_range :name name }))

(defn fake-quant-with-min-max-vars-per-channel 
  "Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

  `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
  to 'outputs' tensor of same shape as `inputs`.

  `[min; max]` define the clamping range for the `inputs` data.
  `inputs` values are quantized into the quantization range (`[0; 2^num_bits - 1]`
  when `narrow_range` is false and `[1; 2^num_bits - 1]` when it is true) and
  then de-quantized and output as floats in `[min; max]` interval.
  `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

  Before quantization, `min` and `max` values are adjusted with the following
  logic.
  It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
  the behavior can be unexpected:
  If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
  If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
  If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
  `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.

  This operation has a gradient and thus allows for training `min` and `max`
  values.

  Args:
    inputs: A `Tensor` of type `float32`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [inputs min max & {:keys [num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_vars_per_channel" [inputs min max] {:num_bits num_bits :narrow_range narrow_range :name name }))

(defn fake-quant-with-min-max-vars-per-channel-gradient 
  "Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
      shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
        same as `gradients`.
      min, max: Quantization interval, floats of shape `[d]`.
    min: A `Tensor` of type `float32`.
    max: A `Tensor` of type `float32`.
    num_bits: An optional `int`. Defaults to `8`.
      The bitwidth of the quantization; between 2 and 16, inclusive.
    narrow_range: An optional `bool`. Defaults to `False`.
      Whether to quantize into 2^num_bits - 1 distinct values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

    backprops_wrt_input: A `Tensor` of type `float32`.
    backprop_wrt_min: A `Tensor` of type `float32`.
    backprop_wrt_max: A `Tensor` of type `float32`.
  "
  [gradients inputs min max & {:keys [num_bits narrow_range name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fake_quant_with_min_max_vars_per_channel_gradient" [gradients inputs min max] {:num_bits num_bits :narrow_range narrow_range :name name }))

(defn quantize 
  "Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.  The
  'round_mode' attribute controls which rounding tie-breaking algorithm is used
  when rounding float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
  if T == qint8: out[i] -= (range(T) + 1) / 2.0
  ```

  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  Assume the input is type float and has a possible range of [0.0, 6.0] and the
  output type is quint8 ([0, 255]). The min_range and max_range values should be
  specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
  value of the input by 255/6 and cast to quint8.

  If the output type was qint8 ([-128, 127]), the operation will additionally
  subtract each value by 128 prior to casting, so that the range of values aligns
  with the range of qint8.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = num_discrete_values / range
  quantized = round(input * range_scale) - round(range_min * range_scale) +
    numeric_limits<T>::min()
  quantized = max(quantized, numeric_limits<T>::min())
  quantized = min(quantized, numeric_limits<T>::max())
  ```

  The biggest difference between this and MIN_COMBINED is that the minimum range
  is rounded first, before it's subtracted from the rounded value. With
  MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
  and dequantizing will introduce a larger and larger error.

  *SCALED mode Example*

  `SCALED` mode matches the quantization approach used in
  `QuantizeAndDequantize{V2|V3}`.

  If the mode is `SCALED`, we do not use the full range of the output type,
  choosing to elide the lowest possible value for symmetry (e.g., output range is
  -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
  0.

  We first find the range of values in our tensor. The
  range we use is always centered on 0, so we find m such that

  ```c++
    m = max(abs(input_min), abs(input_max))
  ```

  Our input tensor range is then `[-m, m]`.

  Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
  If T is signed, this is

  ```
    num_bits = sizeof(T) * 8
    [min_fixed, max_fixed] =
        [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
  ```

  Otherwise, if T is unsigned, the fixed-point range is

  ```
    [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
  ```

  From this we compute our scaling factor, s:

  ```c++
    s = (max_fixed - min_fixed) / (2 * m)
  ```

  Now we can quantize the elements of our tensor:

  ```c++
  result = round(input * s)
  ```

  One thing to watch out for is that the operator may choose to adjust the
  requested minimum and maximum values slightly during the quantization process,
  so you should always use the output ports as the range for further calculations.
  For example, if the requested minimum and maximum values are close to equal,
  they will be separated by a small epsilon value to prevent ill-formed quantized
  buffers from being created. Otherwise, you can end up with buffers where all the
  quantized values map to the same float value, which causes problems for
  operations that have to perform further calculations on them.

  Args:
    input: A `Tensor` of type `float32`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    T: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    mode: An optional `string` from: `\"MIN_COMBINED\", \"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_COMBINED\"`.
    round_mode: An optional `string` from: `\"HALF_AWAY_FROM_ZERO\", \"HALF_TO_EVEN\"`. Defaults to `\"HALF_AWAY_FROM_ZERO\"`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `T`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [input min_range max_range T & {:keys [mode round_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantize" [input min_range max_range T] {:mode mode :round_mode round_mode :name name }))

(defn quantize-and-dequantize 
  "Quantizes then dequantizes a tensor.

  Args:
    input: A `Tensor` to quantize and dequantize.
    input_min: If range_given=True, the minimum input value that needs to be
      represented in the quantized representation.
    input_max: If range_given=True, the maximum input value that needs to be
      represented in the quantized representation.
    signed_input: True if the quantization is signed or unsigned.
    num_bits: The bitwidth of the quantization.
    range_given: If true use `input_min` and `input_max` for the range of the
      input, otherwise determine min and max from the input `Tensor`.
    round_mode: Rounding mode when rounding from float values to quantized ones.
    name: Optional name for the operation.
    narrow_range: If true, then the absolute value of the quantized minimum
      value is the same as the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.

  Returns:
    A `Tensor`. Each element is the result of quantizing and dequantizing the
    corresponding element of `input`.
  "
  [input input_min input_max & {:keys [signed_input num_bits range_given round_mode name narrow_range]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantize_and_dequantize" [input input_min input_max] {:signed_input signed_input :num_bits num_bits :range_given range_given :round_mode round_mode :name name :narrow_range narrow_range }))

(defn quantized-concat 
  "Concatenates quantized tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects with the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    input_mins: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The minimum scalar values for each of the input tensors.
    input_maxes: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The maximum scalar values for each of the input tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `values`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ concat_dim values input_mins input_maxes name ]
  (py/call-attr quantization "quantized_concat"  concat_dim values input_mins input_maxes name ))
