(ns tensorflow.io
  "Public API for tf.io namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow.io"))

(defn decode-and-crop-jpeg 
  "Decode and Crop a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.


  It is equivalent to a combination of decode and crop, but much faster by only
  decoding partial jpeg image.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    crop_window: A `Tensor` of type `int32`.
      1-D.  The crop window: [crop_y, crop_x, crop_height, crop_width].
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    dct_method: An optional `string`. Defaults to `\"\"`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to \"\" which maps to a system-specific
      default.  Currently valid values are [\"INTEGER_FAST\",
      \"INTEGER_ACCURATE\"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [contents crop_window & {:keys [channels ratio fancy_upscaling try_recover_truncated acceptable_fraction dct_method name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_and_crop_jpeg" [contents crop_window] {:channels channels :ratio ratio :fancy_upscaling fancy_upscaling :try_recover_truncated try_recover_truncated :acceptable_fraction acceptable_fraction :dct_method dct_method :name name }))

(defn decode-base64 
  "Decode web-safe base64-encoded strings.

  Input may or may not have padding at the end. See EncodeBase64 for padding.
  Web-safe means that input must use - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Base64 strings to decode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [ input name ]
  (py/call-attr io "decode_base64"  input name ))

(defn decode-bmp 
  "Decode the first frame of a BMP-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the BMP-encoded image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The BMP-encoded image.
    channels: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [contents & {:keys [channels name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_bmp" [contents] {:channels channels :name name }))

(defn decode-compressed 
  "Decompress strings.

  This op decompresses each element of the `bytes` input `Tensor`, which
  is assumed to be compressed using the given `compression_type`.

  The `output` is a string `Tensor` of the same shape as `bytes`,
  each element containing the decompressed data from the corresponding
  element in `bytes`.

  Args:
    bytes: A `Tensor` of type `string`.
      A Tensor of string which is compressed.
    compression_type: An optional `string`. Defaults to `\"\"`.
      A scalar containing either (i) the empty string (no
      compression), (ii) \"ZLIB\", or (iii) \"GZIP\".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [bytes & {:keys [compression_type name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_compressed" [bytes] {:compression_type compression_type :name name }))

(defn decode-csv 
  "Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with specific types.
      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or an empty vector if the column is
      required.
    field_delim: An optional `string`. Defaults to `\",\"`.
      char delimiter to separate fields in a record.
    use_quote_delim: An optional `bool`. Defaults to `True`.
      If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).
    name: A name for the operation (optional).
    na_value: Additional string to recognize as NA/NaN.
    select_cols: Optional sorted list of column indices to select. If specified,
      only this subset of columns will be parsed and returned.

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.

  Raises:
    ValueError: If any of the arguments is malformed.
  "
  [records record_defaults & {:keys [field_delim use_quote_delim name na_value select_cols]
                       :or {name None select_cols None}} ]
    (py/call-attr-kw io "decode_csv" [records record_defaults] {:field_delim field_delim :use_quote_delim use_quote_delim :name name :na_value na_value :select_cols select_cols }))

(defn decode-gif 
  "Decode the frame(s) of a GIF-encoded image to a uint8 tensor.

  GIF images with frame or transparency compression are not supported.
  On Linux and MacOS systems, convert animated GIFs from compressed to
  uncompressed by running:

      convert $src.gif -coalesce $dst.gif

  This op also supports decoding JPEGs and PNGs, though it is cleaner to use
  `tf.image.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The GIF-encoded image.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [ contents name ]
  (py/call-attr io "decode_gif"  contents name ))

(defn decode-image 
  "Function for `decode_bmp`, `decode_gif`, `decode_jpeg`, and `decode_png`.

  Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the
  appropriate operation to convert the input bytes `string` into a `Tensor`
  of type `dtype`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as
  opposed to `decode_bmp`, `decode_jpeg` and `decode_png`, which return 3-D
  arrays `[height, width, num_channels]`. Make sure to take this into account
  when constructing your graph if you are intermixing GIF files with BMP, JPEG,
  and/or PNG files. Alternately, set the `expand_animations` argument of this
  function to `False`, in which case the op will return 3-dimensional tensors
  and will truncate animated GIF files to the first frame.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: An optional `int`. Defaults to `0`. Number of color channels for
      the decoded image.
    dtype: The desired DType of the returned `Tensor`.
    name: A name for the operation (optional)
    expand_animations: Controls the shape of the returned op's output. If
      `True`, the returned op will produce a 3-D tensor for PNG, JPEG, and BMP
      files; and a 4-D tensor for all GIFs, whether animated or not. If,
      `False`, the returned op will produce a 3-D tensor for all file types and
      will truncate animated GIFs to the first frame.

  Returns:
    `Tensor` with type `dtype` and a 3- or 4-dimensional shape, depending on
    the file type and the value of the `expand_animations` parameter.

  Raises:
    ValueError: On incorrect number of channels.
  "
  [contents channels & {:keys [dtype name expand_animations]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_image" [contents channels] {:dtype dtype :name name :expand_animations expand_animations }))

(defn decode-jpeg 
  "Decode a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.


  This op also supports decoding PNGs and non-animated GIFs since the interface is
  the same, though it is cleaner to use `tf.image.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    dct_method: An optional `string`. Defaults to `\"\"`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to \"\" which maps to a system-specific
      default.  Currently valid values are [\"INTEGER_FAST\",
      \"INTEGER_ACCURATE\"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [contents & {:keys [channels ratio fancy_upscaling try_recover_truncated acceptable_fraction dct_method name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_jpeg" [contents] {:channels channels :ratio ratio :fancy_upscaling fancy_upscaling :try_recover_truncated try_recover_truncated :acceptable_fraction acceptable_fraction :dct_method dct_method :name name }))

(defn decode-json-example 
  "Convert JSON-encoded Example records to binary protocol buffer strings.

  This op translates a tensor containing Example records, encoded using
  the [standard JSON
  mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
  into a tensor containing the same records encoded as binary protocol
  buffers. The resulting tensor can then be fed to any of the other
  Example-parsing ops.

  Args:
    json_examples: A `Tensor` of type `string`.
      Each string is a JSON object serialized according to the JSON
      mapping of the Example proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [ json_examples name ]
  (py/call-attr io "decode_json_example"  json_examples name ))

(defn decode-png 
  "Decode a PNG-encoded image to a uint8 or uint16 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the PNG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  If needed, the PNG-encoded image is transformed to match the requested number
  of color channels.

  This op also supports decoding JPEGs and non-animated GIFs since the interface
  is the same, though it is cleaner to use `tf.image.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    dtype: An optional `tf.DType` from: `tf.uint8, tf.uint16`. Defaults to `tf.uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  "
  [contents & {:keys [channels dtype name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_png" [contents] {:channels channels :dtype dtype :name name }))

(defn decode-proto 
  "The op extracts fields from a serialized protocol buffers message into tensors.

  The `decode_proto` op extracts fields from a serialized protocol buffers
  message into tensors.  The fields in `field_names` are decoded and converted
  to the corresponding `output_types` if possible.

  A `message_type` name must be provided to give context for the field names.
  The actual message descriptor can be looked up either in the linked-in
  descriptor pool or a filename provided by the caller using the
  `descriptor_source` attribute.

  Each output tensor is a dense tensor. This means that it is padded to hold
  the largest number of repeated elements seen in the input minibatch. (The
  shape is also padded by one to prevent zero-sized dimensions). The actual
  repeat counts for each example in the minibatch can be found in the `sizes`
  output. In many cases the output of `decode_proto` is fed immediately into
  tf.squeeze if missing values are not a concern. When using tf.squeeze, always
  pass the squeeze dimension explicitly to avoid surprises.

  For the most part, the mapping between Proto field types and TensorFlow dtypes
  is straightforward. However, there are a few special cases:

  - A proto field that contains a submessage or group can only be converted
  to `DT_STRING` (the serialized submessage). This is to reduce the complexity
  of the API. The resulting string can be used as input to another instance of
  the decode_proto op.

  - TensorFlow lacks support for unsigned integers. The ops represent uint64
  types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
  way). Unsigned int32 values can be represented exactly by specifying type
  `DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
  the `output_types` attribute.

  Both binary and text proto serializations are supported, and can be
  chosen using the `format` attribute.

  The `descriptor_source` attribute selects the source of protocol
  descriptors to consult when looking up `message_type`. This may be:

  - An empty string  or \"local://\", in which case protocol descriptors are
  created for C++ (not Python) proto definitions linked to the binary.

  - A file, in which case protocol descriptors are created from the file,
  which is expected to contain a `FileDescriptorSet` serialized as a string.
  NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
  and `--include_imports` options to the protocol compiler `protoc`.

  - A \"bytes://<bytes>\", in which protocol descriptors are created from `<bytes>`,
  which is expected to be a `FileDescriptorSet` serialized as a string.

  Args:
    bytes: A `Tensor` of type `string`.
      Tensor of serialized protos with shape `batch_shape`.
    message_type: A `string`. Name of the proto message type to decode.
    field_names: A list of `strings`.
      List of strings containing proto field names. An extension field can be decoded
      by using its full name, e.g. EXT_PACKAGE.EXT_FIELD_NAME.
    output_types: A list of `tf.DTypes`.
      List of TF types to use for the respective field in field_names.
    descriptor_source: An optional `string`. Defaults to `\"local://\"`.
      Either the special value `local://` or a path to a file containing
      a serialized `FileDescriptorSet`.
    message_format: An optional `string`. Defaults to `\"binary\"`.
      Either `binary` or `text`.
    sanitize: An optional `bool`. Defaults to `False`.
      Whether to sanitize the result or not.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sizes, values).

    sizes: A `Tensor` of type `int32`.
    values: A list of `Tensor` objects of type `output_types`.
  "
  [bytes message_type field_names output_types & {:keys [descriptor_source message_format sanitize name]
                       :or {name None}} ]
    (py/call-attr-kw io "decode_proto" [bytes message_type field_names output_types] {:descriptor_source descriptor_source :message_format message_format :sanitize sanitize :name name }))

(defn decode-raw 
  "Convert raw byte strings into tensors. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(bytes)`. They will be removed in a future version.
Instructions for updating:
bytes is deprecated, use input_bytes instead

Args:
  input_bytes:
    Each element of the input Tensor is converted to an array of bytes.
  out_type:
    `DType` of the output. Acceptable types are `half`, `float`, `double`,
    `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
  little_endian:
    Whether the `input_bytes` data is in little-endian format. Data will be
    converted into host byte order if necessary.
  name: A name for the operation (optional).
  bytes: Deprecated parameter. Use `input_bytes` instead.

Returns:
  A `Tensor` object storing the decoded bytes."
  [input_bytes out_type & {:keys [little_endian name bytes]
                       :or {name None bytes None}} ]
    (py/call-attr-kw io "decode_raw" [input_bytes out_type] {:little_endian little_endian :name name :bytes bytes }))

(defn deserialize-many-sparse 
  "Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `serialize_sparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse.reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: 2-D `Tensor` of type `string` of shape `[N, 3]`.
      The serialized and packed `SparseTensor` objects.
    dtype: The `dtype` of the serialized `SparseTensor` objects.
    rank: (optional) Python int, the rank of the `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  "
  [ serialized_sparse dtype rank name ]
  (py/call-attr io "deserialize_many_sparse"  serialized_sparse dtype rank name ))

(defn encode-base64 
  "Encode strings into web-safe base64 format.

  Refer to the following article for more information on base64 format:
  en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
  end so that the encoded has length multiple of 4. See Padding section of the
  link above.

  Web-safe means that the encoder uses - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Strings to be encoded.
    pad: An optional `bool`. Defaults to `False`.
      Bool whether padding is applied at the ends.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [input & {:keys [pad name]
                       :or {name None}} ]
    (py/call-attr-kw io "encode_base64" [input] {:pad pad :name name }))

(defn encode-jpeg 
  "JPEG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

  The attr `format` can be used to override the color format of the encoded
  output.  Values can be:

  *   `''`: Use a default format based on the number of channels in the image.
  *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
      of `image` must be 1.
  *   `rgb`: Output an RGB JPEG image. The `channels` dimension
      of `image` must be 3.

  If `format` is not specified or is the empty string, a default format is picked
  in function of the number of channels in `image`:

  *   1: Output a grayscale image.
  *   3: Output an RGB image.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    format: An optional `string` from: `\"\", \"grayscale\", \"rgb\"`. Defaults to `\"\"`.
      Per pixel image format.
    quality: An optional `int`. Defaults to `95`.
      Quality of the compression from 0 to 100 (higher is better and slower).
    progressive: An optional `bool`. Defaults to `False`.
      If True, create a JPEG that loads progressively (coarse to fine).
    optimize_size: An optional `bool`. Defaults to `False`.
      If True, spend CPU/RAM to reduce size with no quality change.
    chroma_downsampling: An optional `bool`. Defaults to `True`.
      See http://en.wikipedia.org/wiki/Chroma_subsampling.
    density_unit: An optional `string` from: `\"in\", \"cm\"`. Defaults to `\"in\"`.
      Unit used to specify `x_density` and `y_density`:
      pixels per inch (`'in'`) or centimeter (`'cm'`).
    x_density: An optional `int`. Defaults to `300`.
      Horizontal pixels per density unit.
    y_density: An optional `int`. Defaults to `300`.
      Vertical pixels per density unit.
    xmp_metadata: An optional `string`. Defaults to `\"\"`.
      If not empty, embed this XMP metadata in the image header.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [image & {:keys [format quality progressive optimize_size chroma_downsampling density_unit x_density y_density xmp_metadata name]
                       :or {name None}} ]
    (py/call-attr-kw io "encode_jpeg" [image] {:format format :quality quality :progressive progressive :optimize_size optimize_size :chroma_downsampling chroma_downsampling :density_unit density_unit :x_density x_density :y_density y_density :xmp_metadata xmp_metadata :name name }))

(defn encode-proto 
  "The op serializes protobuf messages provided in the input tensors.

  The types of the tensors in `values` must match the schema for the fields
  specified in `field_names`. All the tensors in `values` must have a common
  shape prefix, *batch_shape*.

  The `sizes` tensor specifies repeat counts for each field.  The repeat count
  (last dimension) of a each tensor in `values` must be greater than or equal
  to corresponding repeat count in `sizes`.

  A `message_type` name must be provided to give context for the field names.
  The actual message descriptor can be looked up either in the linked-in
  descriptor pool or a filename provided by the caller using the
  `descriptor_source` attribute.

  For the most part, the mapping between Proto field types and TensorFlow dtypes
  is straightforward. However, there are a few special cases:

  - A proto field that contains a submessage or group can only be converted
  to `DT_STRING` (the serialized submessage). This is to reduce the complexity
  of the API. The resulting string can be used as input to another instance of
  the decode_proto op.

  - TensorFlow lacks support for unsigned integers. The ops represent uint64
  types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
  way). Unsigned int32 values can be represented exactly by specifying type
  `DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
  the `output_types` attribute.

  The `descriptor_source` attribute selects the source of protocol
  descriptors to consult when looking up `message_type`. This may be:

  - An empty string  or \"local://\", in which case protocol descriptors are
  created for C++ (not Python) proto definitions linked to the binary.

  - A file, in which case protocol descriptors are created from the file,
  which is expected to contain a `FileDescriptorSet` serialized as a string.
  NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
  and `--include_imports` options to the protocol compiler `protoc`.

  - A \"bytes://<bytes>\", in which protocol descriptors are created from `<bytes>`,
  which is expected to be a `FileDescriptorSet` serialized as a string.

  Args:
    sizes: A `Tensor` of type `int32`.
      Tensor of int32 with shape `[batch_shape, len(field_names)]`.
    values: A list of `Tensor` objects.
      List of tensors containing values for the corresponding field.
    field_names: A list of `strings`.
      List of strings containing proto field names.
    message_type: A `string`. Name of the proto message type to decode.
    descriptor_source: An optional `string`. Defaults to `\"local://\"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [sizes values field_names message_type & {:keys [descriptor_source name]
                       :or {name None}} ]
    (py/call-attr-kw io "encode_proto" [sizes values field_names message_type] {:descriptor_source descriptor_source :name name }))

(defn extract-jpeg-shape 
  "Extract the shape information of a JPEG-encoded image.

  This op only parses the image header, so it is much faster than DecodeJpeg.

  Args:
    contents: A `Tensor` of type `string`. 0-D. The JPEG-encoded image.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
      (Optional) The output type of the operation (int32 or int64).
      Defaults to int32.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  "
  [contents & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw io "extract_jpeg_shape" [contents] {:output_type output_type :name name }))

(defn is-jpeg 
  "Convenience function to check if the 'contents' encodes a JPEG image.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    name: A name for the operation (optional)

  Returns:
     A scalar boolean tensor indicating if 'contents' may be a JPEG image.
     is_jpeg is susceptible to false positives.
  "
  [ contents name ]
  (py/call-attr io "is_jpeg"  contents name ))

(defn match-filenames-once 
  "Save the list of files matching pattern, so it is only computed once.

  NOTE: The order of the files returned is deterministic.

  Args:
    pattern: A file pattern (glob), or 1D tensor of file patterns.
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching the pattern(s).
  "
  [ pattern name ]
  (py/call-attr io "match_filenames_once"  pattern name ))

(defn matching-files 
  "Returns the set of files matching one or more glob patterns.

  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.
  Note also that the order of filenames returned is deterministic.

  Args:
    pattern: A `Tensor` of type `string`.
      Shell wildcard pattern(s). Scalar or vector of type string.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [ pattern name ]
  (py/call-attr io "matching_files"  pattern name ))

(defn parse-example 
  "Parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`. We refer to `serialized` as a batch with
  `batch_size` many entries of individual `Example` protos.

  `example_names` may contain descriptive names for the corresponding serialized
  protos. These may be useful for debugging purposes, but they have no effect on
  the output. If not `None`, `example_names` must be the same length as
  `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
  `SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
  and `SparseFeature` is mapped to a `SparseTensor`, and each
  `FixedLenFeature` is mapped to a `Tensor`.

  Each `VarLenFeature` maps to a `SparseTensor` of the specified type
  representing a ragged matrix. Its indices are `[batch, index]` where `batch`
  identifies the example in `serialized`, and `index` is the value's index in
  the list of values associated with that feature and example.

  Each `SparseFeature` maps to a `SparseTensor` of the specified type
  representing a Tensor of `dense_shape` `[batch_size] + SparseFeature.size`.
  Its `values` come from the feature in the examples with key `value_key`.
  A `values[i]` comes from a position `k` in the feature of an example at batch
  entry `batch`. This positional information is recorded in `indices[i]` as
  `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of
  the feature in the example at with key `SparseFeature.index_key[j]`.
  In other words, we split the indices (except the first index indicating the
  batch entry) of a `SparseTensor` by dimension into different features of the
  `Example`. Due to its complexity a `VarLenFeature` should be preferred over a
  `SparseFeature` whenever possible.

  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
  `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

  `FixedLenFeature` entries with a `default_value` are optional. With no default
  value, we will fail if that `Feature` is missing from any example in
  `serialized`.

  Each `FixedLenSequenceFeature` `df` maps to a `Tensor` of the specified type
  (or `tf.float32` if not specified) and shape
  `(serialized.size(), None) + df.shape`.
  All examples in `serialized` will be padded with `default_value` along the
  second dimension.

  Examples:

  For example, if one expects a `tf.float32` `VarLenFeature` `ft` and three
  serialized `Example`s are provided:

  ```
  serialized = [
    features
      { feature { key: \"ft\" value { float_list { value: [1.0, 2.0] } } } },
    features
      { feature []},
    features
      { feature { key: \"ft\" value { float_list { value: [3.0] } } }
  ]
  ```

  then the output will look like:

  ```python
  {\"ft\": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      dense_shape=(3, 2)) }
  ```

  If instead a `FixedLenSequenceFeature` with `default_value = -1.0` and
  `shape=[]` is used then the output will look like:

  ```python
  {\"ft\": [[1.0, 2.0], [3.0, -1.0]]}
  ```

  Given two `Example` input protos in `serialized`:

  ```
  [
    features {
      feature { key: \"kw\" value { bytes_list { value: [ \"knit\", \"big\" ] } } }
      feature { key: \"gps\" value { float_list { value: [] } } }
    },
    features {
      feature { key: \"kw\" value { bytes_list { value: [ \"emmy\" ] } } }
      feature { key: \"dank\" value { int64_list { value: [ 42 ] } } }
      feature { key: \"gps\" value { } }
    }
  ]
  ```

  And arguments

  ```
  example_names: [\"input0\", \"input1\"],
  features: {
      \"kw\": VarLenFeature(tf.string),
      \"dank\": VarLenFeature(tf.int64),
      \"gps\": VarLenFeature(tf.float32),
  }
  ```

  Then the output is a dictionary:

  ```python
  {
    \"kw\": SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[\"knit\", \"big\", \"emmy\"]
        dense_shape=[2, 2]),
    \"dank\": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        dense_shape=[2, 1]),
    \"gps\": SparseTensor(
        indices=[],
        values=[],
        dense_shape=[2, 0]),
  }
  ```

  For dense results in two serialized `Example`s:

  ```
  [
    features {
      feature { key: \"age\" value { int64_list { value: [ 0 ] } } }
      feature { key: \"gender\" value { bytes_list { value: [ \"f\" ] } } }
     },
     features {
      feature { key: \"age\" value { int64_list { value: [] } } }
      feature { key: \"gender\" value { bytes_list { value: [ \"f\" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  example_names: [\"input0\", \"input1\"],
  features: {
      \"age\": FixedLenFeature([], dtype=tf.int64, default_value=-1),
      \"gender\": FixedLenFeature([], dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    \"age\": [[0], [-1]],
    \"gender\": [[\"f\"], [\"f\"]],
  }
  ```

  An alternative to `VarLenFeature` to obtain a `SparseTensor` is
  `SparseFeature`. For example, given two `Example` input protos in
  `serialized`:

  ```
  [
    features {
      feature { key: \"val\" value { float_list { value: [ 0.5, -1.0 ] } } }
      feature { key: \"ix\" value { int64_list { value: [ 3, 20 ] } } }
    },
    features {
      feature { key: \"val\" value { float_list { value: [ 0.0 ] } } }
      feature { key: \"ix\" value { int64_list { value: [ 42 ] } } }
    }
  ]
  ```

  And arguments

  ```
  example_names: [\"input0\", \"input1\"],
  features: {
      \"sparse\": SparseFeature(
          index_key=\"ix\", value_key=\"val\", dtype=tf.float32, size=100),
  }
  ```

  Then the output is a dictionary:

  ```python
  {
    \"sparse\": SparseTensor(
        indices=[[0, 3], [0, 20], [1, 42]],
        values=[0.5, -1.0, 0.0]
        dense_shape=[2, 100]),
  }
  ```

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A `dict` mapping feature keys to `FixedLenFeature`,
      `VarLenFeature`, and `SparseFeature` values.
    name: A name for this operation (optional).
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized features name example_names ]
  (py/call-attr io "parse_example"  serialized features name example_names ))

(defn parse-sequence-example 
  "Parses a batch of `SequenceExample` protos.

  Parses a vector of serialized
  [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`.

  This op parses serialized sequence examples into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature` and `FixedLenFeature` objects.
  Each `VarLenFeature` is mapped to a `SparseTensor`, and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature` and `FixedLenSequenceFeature`
  objects. Each `VarLenFeature` is mapped to a `SparseTensor`, and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(B,T,) + df.dense_shape` for `FixedLenSequenceFeature`
  `df`, where `B` is the batch size, and `T` is the length of the associated
  `FeatureList` in the `SequenceExample`. For instance,
  `FixedLenSequenceFeature([])` yields a scalar 2-D `Tensor` of static shape
  `[None, None]` and dynamic shape `[B, T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 3-D matrix `Tensor`
  of static shape `[None, None, k]` and dynamic shape `[B, T, k]`.

  Like the input, the resulting output tensors have a batch dimension. This
  means that the original per-example shapes of `VarLenFeature`s and
  `FixedLenSequenceFeature`s can be lost. To handle that situation, this op also
  provides dicts of shape tensors as part of the output. There is one dict for
  the context features, and one for the feature_list features. Context features
  of type `FixedLenFeature`s will not be present, since their shapes are already
  known by the caller. In situations where the input 'FixedLenFeature`s are of
  different lengths across examples, the shorter examples will be padded with
  default datatype values: 0 for numeric types, and the empty string for string
  types.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Args:
    serialized: A vector (1-D Tensor) of type string containing binary
      serialized `SequenceExample` protos.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. These features are associated with a
      `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
      associated with data within the `FeatureList` section of the
      `SequenceExample` proto.
    example_names: A vector (1-D Tensor) of strings (optional), the name of the
      serialized protos.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s and
    `SparseTensor`s. The first dict contains the context key/values,
    the second dict contains the feature_list key/values, and the final dict
    contains the lengths of any dense feature_list features.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized context_features sequence_features example_names name ]
  (py/call-attr io "parse_sequence_example"  serialized context_features sequence_features example_names name ))

(defn parse-single-example 
  "Parses a single `Example` proto.

  Similar to `parse_example`, except:

  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.

  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).

  One might see performance advantages by batching `Example` protos with
  `parse_example` instead of using this function directly.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
      See `_parse_single_example_raw` documentation for more details.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.
      See `_parse_single_example_raw` documentation for more details.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized features name example_names ]
  (py/call-attr io "parse_single_example"  serialized features name example_names ))

(defn parse-single-sequence-example 
  "Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialized sequence example into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature` and `FixedLenFeature` objects.
  Each `VarLenFeature` is mapped to a `SparseTensor`, and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature` and `FixedLenSequenceFeature`
  objects. Each `VarLenFeature` is mapped to a `SparseTensor`, and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`, where
  `T` is the length of the associated `FeatureList` in the `SequenceExample`.
  For instance, `FixedLenSequenceFeature([])` yields a scalar 1-D `Tensor` of
  static shape `[None]` and dynamic shape `[T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 2-D matrix `Tensor`
  of static shape `[None, k]` and dynamic shape `[T, k]`.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Note that the batch version of this function, `tf.parse_sequence_example`,
  is written for better memory efficiency and will be faster on large
  `SequenceExample`s.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. These features are associated with a
      `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
      associated with data within the `FeatureList` section of the
      `SequenceExample` proto.
    example_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    ValueError: if any feature is invalid.
  "
  [ serialized context_features sequence_features example_name name ]
  (py/call-attr io "parse_single_sequence_example"  serialized context_features sequence_features example_name name ))

(defn parse-tensor 
  "Transforms a serialized tensorflow.TensorProto proto into a Tensor.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar string containing a serialized TensorProto proto.
    out_type: A `tf.DType`.
      The type of the serialized tensor.  The provided type must match the
      type of the serialized tensor and no implicit conversion will take place.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  "
  [ serialized out_type name ]
  (py/call-attr io "parse_tensor"  serialized out_type name ))

(defn read-file 
  "Reads and outputs the entire contents of the input filename.

  Args:
    filename: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [ filename name ]
  (py/call-attr io "read_file"  filename name ))
(defn serialize-many-sparse 
  "Serialize `N`-minibatch `SparseTensor` into an `[N, 3]` `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.

  Returns:
    A matrix (2-D `Tensor`) with `N` rows and `3` columns. Each column
    represents serialized `SparseTensor`'s indices, values, and shape
    (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  "
  [sp_input name  & {:keys [out_type]} ]
    (py/call-attr-kw io "serialize_many_sparse" [sp_input name] {:out_type out_type }))
(defn serialize-sparse 
  "Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.

  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  "
  [sp_input name  & {:keys [out_type]} ]
    (py/call-attr-kw io "serialize_sparse" [sp_input name] {:out_type out_type }))

(defn serialize-tensor 
  "Transforms a Tensor into a serialized TensorProto proto.

  Args:
    tensor: A `Tensor`. A Tensor of type `T`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [ tensor name ]
  (py/call-attr io "serialize_tensor"  tensor name ))

(defn tf-record-iterator 
  "An iterator that read the records from a TFRecords file. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use eager execution and: 
`tf.data.TFRecordDataset(path)`

Args:
  path: The path to the TFRecords file.
  options: (optional) A TFRecordOptions object.

Yields:
  Strings.

Raises:
  IOError: If `path` cannot be opened for reading."
  [ path options ]
  (py/call-attr io "tf_record_iterator"  path options ))

(defn write-file 
  "Writes contents to the file at input filename. Creates file and recursively

  creates directory if not existing.

  Args:
    filename: A `Tensor` of type `string`.
      scalar. The name of the file to which we write the contents.
    contents: A `Tensor` of type `string`.
      scalar. The content to be written to the output file.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ filename contents name ]
  (py/call-attr io "write_file"  filename contents name ))
(defn write-graph 
  "Writes a graph proto to a file.

  The graph is written as a text proto unless `as_text` is `False`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  or

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_or_graph_def: A `Graph` or a `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.

  Returns:
    The path of the output proto file.
  "
  [graph_or_graph_def logdir name  & {:keys [as_text]} ]
    (py/call-attr-kw io "write_graph" [graph_or_graph_def logdir name] {:as_text as_text }))
