(ns tensorflow.-api.v1.compat.v2.image
  "Image processing and decoding ops.

See the [Images](https://tensorflow.org/api_guides/python/image) guide.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow._api.v1.compat.v2.image"))

(defn adjust-brightness 
  "Adjust the brightness of RGB or Grayscale images.

  This is a convenience method that converts RGB images to float
  representation, adjusts their brightness, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.

  The value `delta` is added to all components of the tensor `image`. `image` is
  converted to `float` and scaled appropriately if it is in fixed-point
  representation, and `delta` is converted to the same data type. For regular
  images, `delta` should be in the range `[0,1)`, as it is added to the image in
  floating point representation, where pixel values are in the `[0,1)` range.

  Args:
    image: RGB image or images to adjust.
    delta: A scalar. Amount to add to the pixel values.

  Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.
  
  Usage Example:
    ```python
    import tensorflow as tf
    x = tf.random.normal(shape=(256, 256, 3))
    tf.image.adjust_brightness(x, delta=0.1)
    ```
  "
  [ image delta ]
  (py/call-attr image "adjust_brightness"  image delta ))

(defn adjust-contrast 
  "Adjust contrast of RGB or grayscale images.

  This is a convenience method that converts RGB images to float
  representation, adjusts their contrast, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, this Op computes the mean of the image pixels in the
  channel and then adjusts each component `x` of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Args:
    images: Images to adjust.  At least 3-D.
    contrast_factor: A float multiplier for adjusting contrast.

  Returns:
    The contrast-adjusted image or images.
    
  Usage Example:
    ```python
    import tensorflow as tf
    x = tf.random.normal(shape=(256, 256, 3))
    tf.image.adjust_contrast(x,2)
    ```
  "
  [ images contrast_factor ]
  (py/call-attr image "adjust_contrast"  images contrast_factor ))
(defn adjust-gamma 
  "Performs Gamma Correction on the input image.

  Also known as Power Law Transform. This function converts the
  input images at first to float representation, then transforms them
  pixelwise according to the equation `Out = gain * In**gamma`,
  and then converts the back to the original data type.

  Args:
    image : RGB image or images to adjust.
    gamma : A scalar or tensor. Non negative real number.
    gain  : A scalar or tensor. The constant multiplier.

  Returns:
    A Tensor. A Gamma-adjusted tensor of the same shape and type as `image`.
  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_gamma(x, 0.2)
    ```
  Raises:
    ValueError: If gamma is negative.
  Notes:
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.
  References:
    [1] http://en.wikipedia.org/wiki/Gamma_correction
  "
  [image  & {:keys [gamma gain]} ]
    (py/call-attr-kw image "adjust_gamma" [image] {:gamma gamma :gain gain }))

(defn adjust-hue 
  "Adjust hue of RGB images.

  This is a convenience method that converts an RGB image to float
  representation, converts it to HSV, add an offset to the hue channel, converts
  back to RGB and then back to the original data type. If several adjustments
  are chained it is advisable to minimize the number of redundant conversions.

  `image` is an RGB image.  The image hue is adjusted by converting the
  image(s) to HSV and rotating the hue channel (H) by
  `delta`.  The image is then converted back to RGB.

  `delta` must be in the interval `[-1, 1]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    delta: float.  How much to add to the hue channel.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_hue(x, 0.2)
    ```
  "
  [ image delta name ]
  (py/call-attr image "adjust_hue"  image delta name ))

(defn adjust-jpeg-quality 
  "Adjust jpeg encoding quality of an RGB image.

  This is a convenience method that adjusts jpeg encoding quality of an
  RGB image.

  `image` is an RGB image.  The image's encoding quality is adjusted
  to `jpeg_quality`.
  `jpeg_quality` must be in the interval `[0, 100]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    jpeg_quality: Python int or Tensor of type int32.  jpeg encoding quality.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.
  
  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_jpeg_quality(x, 75)
    ```
  Raises:
    InvalidArgumentError: quality must be in [0,100]
    InvalidArgumentError: image must have 1 or 3 channels
  "
  [ image jpeg_quality name ]
  (py/call-attr image "adjust_jpeg_quality"  image jpeg_quality name ))

(defn adjust-saturation 
  "Adjust saturation of RGB images.

  This is a convenience method that converts RGB images to float
  representation, converts them to HSV, add an offset to the saturation channel,
  converts back to RGB and then back to the original data type. If several
  adjustments are chained it is advisable to minimize the number of redundant
  conversions.

  `image` is an RGB image or images.  The image saturation is adjusted by
  converting the images to HSV and multiplying the saturation (S) channel by
  `saturation_factor` and clipping. The images are then converted back to RGB.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    saturation_factor: float. Factor to multiply the saturation by.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.
    
  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3))
    >> tf.image.adjust_saturation(x, 0.5)
    ```
    
  Raises:
    InvalidArgumentError: input must have 3 channels
  "
  [ image saturation_factor name ]
  (py/call-attr image "adjust_saturation"  image saturation_factor name ))

(defn central-crop 
  "Crop the central region of the image(s).

  Remove the outer parts of an image but retain the central region of the image
  along each dimension. If we specify central_fraction = 0.5, this function
  returns the region marked with \"X\" in the below diagram.

       --------
      |        |
      |  XXXX  |
      |  XXXX  |
      |        |   where \"X\" is the central 50% of the image.
       --------

  This function works on either a single image (`image` is a 3-D Tensor), or a
  batch of images (`image` is a 4-D Tensor).

  Args:
    image: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D
      Tensor of shape [batch_size, height, width, depth].
    central_fraction: float (0, 1], fraction of size to crop
  Usage Example: ```python >> import tensorflow as tf >> x =
    tf.random.normal(shape=(256, 256, 3)) >> tf.image.central_crop(x, 0.5) ```

  Raises:
    ValueError: if central_crop_fraction is not within (0, 1].

  Returns:
    3-D / 4-D float Tensor, as per the input.
  "
  [ image central_fraction ]
  (py/call-attr image "central_crop"  image central_fraction ))

(defn combined-non-max-suppression 
  "Greedily selects a subset of bounding boxes in descending order of score.

  This operation performs non_max_suppression on the inputs per batch, across
  all classes.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system. Also note that
  this algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is the final boxes, scores and classes tensor
  returned after performing non_max_suppression.

  Args:
    boxes: A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q`
      is 1 then same boxes are used for all classes otherwise, if `q` is equal
      to number of classes, class-specific boxes are used.
    scores: A 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`
      representing a single score corresponding to each box (each row of boxes).
    max_output_size_per_class: A scalar integer `Tensor` representing the
      maximum number of boxes to be selected by non max suppression per class
    max_total_size: A scalar representing maximum number of boxes retained over
      all classes.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_per_class: If false, the output nmsed boxes, scores and classes are
      padded/clipped to `max_total_size`. If true, the output nmsed boxes,
      scores and classes are padded to be of length
      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
      which case it is clipped to `max_total_size`. Defaults to false.
    clip_boxes: If true, the coordinates of output nmsed boxes will be clipped
      to [0, 1]. If false, output the box coordinates as it is. Defaults to
      true.
    name: A name for the operation (optional).

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'valid_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top valid_detections[i] entries
      in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
  "
  [boxes scores max_output_size_per_class max_total_size & {:keys [iou_threshold score_threshold pad_per_class clip_boxes name]
                       :or {name None}} ]
    (py/call-attr-kw image "combined_non_max_suppression" [boxes scores max_output_size_per_class max_total_size] {:iou_threshold iou_threshold :score_threshold score_threshold :pad_per_class pad_per_class :clip_boxes clip_boxes :name name }))

(defn convert-image-dtype 
  "Convert `image` to `dtype`, scaling its values if needed.

  Images that are represented using floating point values are expected to have
  values in the range [0,1). Image data stored in integer data types are
  expected to have values in the range `[0,MAX]`, where `MAX` is the largest
  positive representable number for the data type.

  This op converts between data types, scaling the values appropriately before
  casting.

  Note that converting from floating point inputs to integer types may lead to
  over/underflow problems. Set saturate to `True` to avoid such problem in
  problematic conversions. If enabled, saturation will clip the output into the
  allowed range before performing a potentially dangerous cast (and only before
  performing such a cast, i.e., when casting from a floating point to an integer
  type, and when casting from a signed to an unsigned type; `saturate` has no
  effect on casts between floats, or on casts that increase the type's range).

  Args:
    image: An image.
    dtype: A `DType` to convert `image` to.
    saturate: If `True`, clip the input before casting (if necessary).
    name: A name for this operation (optional).

  Returns:
    `image`, converted to `dtype`.

  Usage Example:
    ```python
    >> import tensorflow as tf
    >> x = tf.random.normal(shape=(256, 256, 3), dtype=tf.float32)
    >> tf.image.convert_image_dtype(x, dtype=tf.float16, saturate=False)
    ```

  Raises:
    AttributeError: Raises an attribute error when dtype is neither
    float nor integer
  "
  [image dtype & {:keys [saturate name]
                       :or {name None}} ]
    (py/call-attr-kw image "convert_image_dtype" [image dtype] {:saturate saturate :name name }))

(defn crop-and-resize 
  "Extracts crops from the input image tensor and resizes them.

  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input
  image and does not allow resizing or aspect ratio change.

  Returns a tensor with `crops` from the input `image` at positions defined at
  the bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.compat.v1.image.resize_bilinear()` or
  `tf.compat.v1.image.resize_nearest_neighbor()`(depends on the `method`
  argument) with
  `align_corners=True`.

  Args:
    image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at `y *
      (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1]` in image height coordinates.
      We do allow `y1` > `y2`, in which case the sampled crop is an up-down
      flipped version of the original image. The width dimension is treated
      similarly. Normalized coordinates outside the `[0, 1]` range are allowed,
      in which case we use `extrapolation_value` to extrapolate the input image
      values.
    box_indices: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0,
      batch)`. The value of `box_ind[i]` specifies the image that the `i`-th box
      refers to.
    crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`.
      All cropped image patches are resized to this size. The aspect ratio of
      the image content is not preserved. Both `crop_height` and `crop_width`
      need to be positive.
    method: An optional string specifying the sampling method for resizing. It
      can be either `\"bilinear\"` or `\"nearest\"` and default to `\"bilinear\"`.
      Currently two sampling methods are supported: Bilinear and Nearest
        Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0`. Value used for
      extrapolation, when applicable.
    name: A name for the operation (optional).

  Returns:
    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.

  Example:

  ```python
  import tensorflow as tf
  BATCH_SIZE = 1
  NUM_BOXES = 5
  IMAGE_HEIGHT = 256
  IMAGE_WIDTH = 256
  CHANNELS = 3
  CROP_SIZE = (24, 24)

  image = tf.random.normal(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
  CHANNELS) )
  boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
  box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
  maxval=BATCH_SIZE, dtype=tf.int32)
  output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
  output.shape  #=> (5, 24, 24, 3)
  ```
  "
  [image boxes box_indices crop_size & {:keys [method extrapolation_value name]
                       :or {name None}} ]
    (py/call-attr-kw image "crop_and_resize" [image boxes box_indices crop_size] {:method method :extrapolation_value extrapolation_value :name name }))

(defn crop-to-bounding-box 
  "Crops an image to a specified bounding box.

  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    offset_height: Vertical coordinate of the top-left corner of the result in
      the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
      the input.
    target_height: Height of the result.
    target_width: Width of the result.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative, or either `target_height` or `target_width` is not positive.
  "
  [ image offset_height offset_width target_height target_width ]
  (py/call-attr image "crop_to_bounding_box"  image offset_height offset_width target_height target_width ))

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
    (py/call-attr-kw image "decode_and_crop_jpeg" [contents crop_window] {:channels channels :ratio ratio :fancy_upscaling fancy_upscaling :try_recover_truncated try_recover_truncated :acceptable_fraction acceptable_fraction :dct_method dct_method :name name }))

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
    (py/call-attr-kw image "decode_bmp" [contents] {:channels channels :name name }))

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
  (py/call-attr image "decode_gif"  contents name ))

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
    (py/call-attr-kw image "decode_image" [contents channels] {:dtype dtype :name name :expand_animations expand_animations }))

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
    (py/call-attr-kw image "decode_jpeg" [contents] {:channels channels :ratio ratio :fancy_upscaling fancy_upscaling :try_recover_truncated try_recover_truncated :acceptable_fraction acceptable_fraction :dct_method dct_method :name name }))

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
    (py/call-attr-kw image "decode_png" [contents] {:channels channels :dtype dtype :name name }))

(defn draw-bounding-boxes 
  "Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more
  bounding boxes specified by the locations in `boxes`. The coordinates of the
  each bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`. 3-D with shape `[batch,
      num_bounding_boxes, 4]` containing bounding boxes.
    colors: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle
      through for the boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  "
  [ images boxes colors name ]
  (py/call-attr image "draw_bounding_boxes"  images boxes colors name ))

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
    (py/call-attr-kw image "encode_jpeg" [image] {:format format :quality quality :progressive progressive :optimize_size optimize_size :chroma_downsampling chroma_downsampling :density_unit density_unit :x_density x_density :y_density y_density :xmp_metadata xmp_metadata :name name }))

(defn encode-png 
  "PNG-encode an image.

  `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
  where `channels` is:

  *   1: for grayscale.
  *   2: for grayscale + alpha.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level, generating
  the smallest output, but is slower.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [image & {:keys [compression name]
                       :or {name None}} ]
    (py/call-attr-kw image "encode_png" [image] {:compression compression :name name }))

(defn extract-glimpse 
  "Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`. A 4-D float tensor of shape
      `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`. A 1-D tensor of 2 elements containing the
      size of the glimpses to extract.  The glimpse height must be specified
      first, following by the glimpse width.
    offsets: A `Tensor` of type `float32`. A 2-D integer tensor of shape
      `[batch_size, 2]` containing the y, x locations of the center of each
      window.
    centered: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are centered relative to the image, in which case the (0, 0)
      offset is relative to the center of the input images. If false, the (0,0)
      offset corresponds to the upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`. indicates if the offset
      coordinates are normalized.
    noise: An optional `string`. Defaults to `uniform`. indicates if the noise
      should be `uniform` (uniform distribution), `gaussian` (gaussian
      distribution), or `zero` (zero padding).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.

  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 3
    IMAGE_WIDTH = 3
    CHANNELS = 1
    GLIMPSE_SIZE = (2, 2)
    image = tf.reshape(tf.range(9, delta=1, dtype=tf.float32),
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    output = tf.image.extract_glimpse(image, size=GLIMPSE_SIZE,
      offsets=[[1, 1]], centered=False, normalized=False)
     ```
  "
  [input size offsets & {:keys [centered normalized noise name]
                       :or {name None}} ]
    (py/call-attr-kw image "extract_glimpse" [input size offsets] {:centered centered :normalized normalized :noise noise :name name }))

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
    (py/call-attr-kw image "extract_jpeg_shape" [contents] {:output_type output_type :name name }))

(defn extract-patches 
  "Extract `patches` from `images`.

  This op collects patches from the input image, as if applying a
  convolution. All extracted patches are stacked in the depth (last) dimension
  of the output.

  Specifically, the op extracts patches of shape `sizes` which are `strides`
  apart in the input image. The output is subsampled using the `rates` argument,
  in the same manner as \"atrous\" or \"dilated\" convolutions.

  The result is a 4D tensor which is indexed by batch, row, and column.
  `output[i, x, y]` contains a flattened patch of size `sizes[1], sizes[2]`
  which is taken from the input starting at
  `images[i, x*strides[1], y*strides[2]]`.

  Each output patch can be reshaped to `sizes[1], sizes[2], depth`, where
  `depth` is `images.shape[3]`.

  The output elements are taken from the input at intervals given by the `rate`
  argument, as in dilated convolutions.

  The `padding` argument has no effect on the size of each patch, it determines
  how many patches are extracted. If `VALID`, only patches which are fully
  contained in the input image are included. If `SAME`, all patches whose
  starting point is inside the input are included, and areas outside the input
  default to zero.

  Example:

  ```
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
    images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

    # We generate two outputs as follows:
    # 1. 3x3 patches with stride length 5
    # 2. Same as above, but the rate is increased to 2
    tf.extract_image_patches(images=images,
                             ksizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 1, 1, 1],
                             padding='VALID')

    # Yields:
    [[[[ 1  2  3 11 12 13 21 22 23]
       [ 6  7  8 16 17 18 26 27 28]]
      [[51 52 53 61 62 63 71 72 73]
       [56 57 58 66 67 68 76 77 78]]]]
  ```

  If we mark the pixels in the input image which are taken for the output with
  `*`, we see the pattern:

  ```
     *  *  *  4  5  *  *  *  9 10
     *  *  * 14 15  *  *  * 19 20
     *  *  * 24 25  *  *  * 29 30
    31 32 33 34 35 36 37 38 39 40
    41 42 43 44 45 46 47 48 49 50
     *  *  * 54 55  *  *  * 59 60
     *  *  * 64 65  *  *  * 69 70
     *  *  * 74 75  *  *  * 79 80
    81 82 83 84 85 86 87 88 89 90
    91 92 93 94 95 96 97 98 99 100
  ```

  ```
    tf.extract_image_patches(images=images,
                             sizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 2, 2, 1],
                             padding='VALID')

    # Yields:
    [[[[  1   3   5  21  23  25  41  43  45]
       [  6   8  10  26  28  30  46  48  50]]

      [[ 51  53  55  71  73  75  91  93  95]
       [ 56  58  60  76  78  80  96  98 100]]]]
  ```

  We can again draw the effect, this time using the symbols `*`, `x`, `+` and
  `o` to distinguish the patches:

  ```
     *  2  *  4  *  x  7  x  9  x
    11 12 13 14 15 16 17 18 19 20
     * 22  * 24  *  x 27  x 29  x
    31 32 33 34 35 36 37 38 39 40
     * 42  * 44  *  x 47  x 49  x
     + 52  + 54  +  o 57  o 59  o
    61 62 63 64 65 66 67 68 69 70
     + 72  + 74  +  o 77  o 79  o
    81 82 83 84 85 86 87 88 89 90
     + 92  + 94  +  o 97  o 99  o
  ```

  Args:
    images: A 4-D Tensor with shape `[batch, in_rows, in_cols, depth]
    sizes: The size of the extracted patches. Must
      be [1, size_rows, size_cols, 1].
    strides: A 1-D Tensor of length 4. How far the centers of two consecutive
      patches are in the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    rates: A 1-D Tensor of length 4. Must be: `[1, rate_rows, rate_cols, 1]`.
      This is the input stride, specifying how far two consecutive patch samples
      are in the input. Equivalent to extracting patches with `patch_sizes_eff =
      patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
      them spatially by a factor of `rates`. This is equivalent to `rate` in
      dilated (a.k.a. Atrous) convolutions.
    padding: The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A 4-D Tensor of the same type as the input.
  "
  [ images sizes strides rates padding name ]
  (py/call-attr image "extract_patches"  images sizes strides rates padding name ))

(defn flip-left-right 
  "Flip an image horizontally (left to right).

  Outputs the contents of `image` flipped along the width dimension.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [ image ]
  (py/call-attr image "flip_left_right"  image ))

(defn flip-up-down 
  "Flip an image vertically (upside down).

  Outputs the contents of `image` flipped along the height dimension.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.

  Returns:
    A `Tensor` of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [ image ]
  (py/call-attr image "flip_up_down"  image ))

(defn grayscale-to-rgb 
  "Converts one or more images from Grayscale to RGB.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 3, containing the RGB value of the pixels.
  The input images' last dimension must be size 1.

  Args:
    images: The Grayscale tensor to convert. Last dimension must be size 1.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  "
  [ images name ]
  (py/call-attr image "grayscale_to_rgb"  images name ))

(defn hsv-to-rgb 
  "Convert one or more images from HSV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  See `rgb_to_hsv` for a description of the HSV encoding.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      1-D or higher rank. HSV data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  "
  [ images name ]
  (py/call-attr image "hsv_to_rgb"  images name ))

(defn image-gradients 
  "Returns image gradients (dy, dx) for each color channel.

  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.

  Arguments:
    image: Tensor with shape [batch_size, h, w, d].

  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).
    
  Usage Example:
    ```python
    BATCH_SIZE = 1
    IMAGE_HEIGHT = 5
    IMAGE_WIDTH = 5
    CHANNELS = 1
    image = tf.reshape(tf.range(IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS, 
      delta=1, dtype=tf.float32), 
      shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    dx, dy = tf.image.image_gradients(image)
    print(image[0, :,:,0])
    tf.Tensor(
      [[ 0.  1.  2.  3.  4.]
      [ 5.  6.  7.  8.  9.]
      [10. 11. 12. 13. 14.]
      [15. 16. 17. 18. 19.]
      [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float32)
    print(dx[0, :,:,0])
    tf.Tensor(
      [[5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [5. 5. 5. 5. 5.]
      [0. 0. 0. 0. 0.]], shape=(5, 5), dtype=float32)    
    print(dy[0, :,:,0])
    tf.Tensor(
      [[1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]
      [1. 1. 1. 1. 0.]], shape=(5, 5), dtype=float32)
    ```
    
  Raises:
    ValueError: If `image` is not a 4D tensor.
  "
  [ image ]
  (py/call-attr image "image_gradients"  image ))

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
  (py/call-attr image "is_jpeg"  contents name ))

(defn non-max-suppression 
  "Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
  "
  [boxes scores max_output_size & {:keys [iou_threshold score_threshold name]
                       :or {name None}} ]
    (py/call-attr-kw image "non_max_suppression" [boxes scores max_output_size] {:iou_threshold iou_threshold :score_threshold score_threshold :name name }))

(defn non-max-suppression-overlaps 
  "Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high overlap with previously selected boxes.
  N-by-n overlap values are supplied as square matrix.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices = tf.image.non_max_suppression_overlaps(
        overlaps, scores, max_output_size, iou_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    overlaps: A 2-D float `Tensor` of shape `[num_boxes, num_boxes]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    overlap_threshold: A float representing the threshold for deciding whether
      boxes overlap too much with respect to the provided overlap values.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the overlaps tensor, where `M <= max_output_size`.
  "
  [overlaps scores max_output_size & {:keys [overlap_threshold score_threshold name]
                       :or {name None}} ]
    (py/call-attr-kw image "non_max_suppression_overlaps" [overlaps scores max_output_size] {:overlap_threshold overlap_threshold :score_threshold score_threshold :name name }))

(defn non-max-suppression-padded 
  "Greedily selects a subset of bounding boxes in descending order of score.

  Performs algorithmically equivalent operation to tf.image.non_max_suppression,
  with the addition of an optional parameter which zero-pads the output to
  be of size `max_output_size`.
  The output of this operation is a tuple containing the set of integers
  indexing into the input collection of bounding boxes representing the selected
  boxes and the number of valid indices in the index set.  The bounding box
  coordinates corresponding to the selected indices can then be obtained using
  the `tf.slice` and `tf.gather` operations.  For example:
    ```python
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold,
        score_threshold, pad_to_max_output_size=True)
    selected_indices = tf.slice(
        selected_indices_padded, tf.constant([0]), num_valid)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    pad_to_max_output_size: bool.  If True, size of `selected_indices` output is
      padded to `max_output_size`.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    valid_outputs: A scalar integer `Tensor` denoting how many elements in
    `selected_indices` are valid.  Valid elements occur first, then padding.
  "
  [boxes scores max_output_size & {:keys [iou_threshold score_threshold pad_to_max_output_size name]
                       :or {name None}} ]
    (py/call-attr-kw image "non_max_suppression_padded" [boxes scores max_output_size] {:iou_threshold iou_threshold :score_threshold score_threshold :pad_to_max_output_size pad_to_max_output_size :name name }))

(defn non-max-suppression-with-scores 
  "Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices, selected_scores = tf.image.non_max_suppression_v2(
        boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,
        soft_nms_sigma=0.5)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  This function generalizes the `tf.image.non_max_suppression` op by also
  supporting a Soft-NMS (with Gaussian weighting) mode (c.f.
  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
  of other overlapping boxes instead of directly causing them to be pruned.
  Consequently, in contrast to `tf.image.non_max_suppression`,
  `tf.image.non_max_suppression_v2` returns the new scores of each input box in
  the second output, `selected_scores`.

  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
  larger than 0.  When `soft_nms_sigma` equals 0, the behavior of
  `tf.image.non_max_suppression_v2` is identical to that of
  `tf.image.non_max_suppression` (except for the extra output) both in function
  and in running time.

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
        `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
        NMS.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    selected_scores: A 1-D float tensor of shape `[M]` representing the
      corresponding scores for each selected box, where `M <= max_output_size`.
      Scores only differ from corresponding input scores when using Soft NMS
      (i.e. when `soft_nms_sigma>0`)
  "
  [boxes scores max_output_size & {:keys [iou_threshold score_threshold soft_nms_sigma name]
                       :or {name None}} ]
    (py/call-attr-kw image "non_max_suppression_with_scores" [boxes scores max_output_size] {:iou_threshold iou_threshold :score_threshold score_threshold :soft_nms_sigma soft_nms_sigma :name name }))

(defn pad-to-bounding-box 
  "Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  "
  [ image offset_height offset_width target_height target_width ]
  (py/call-attr image "pad_to_bounding_box"  image offset_height offset_width target_height target_width ))

(defn per-image-standardization 
  "Linearly scales each image in `image` to have mean 0 and variance 1.

  For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,
  where

  - `mean` is the average of all values in `x`
  - `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to
    protect against division by 0 when handling uniform images
    - `N` is the number of elements in `x`
    - `stddev` is the standard deviation of all values in `x`

  Args:
    image: An n-D Tensor with at least 3 dimensions, the last 3 of which are the
      dimensions of each image.

  Returns:
    A `Tensor` with same shape and dtype as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  "
  [ image ]
  (py/call-attr image "per_image_standardization"  image ))

(defn psnr 
  "Returns the Peak Signal-to-Noise Ratio between a and b.

  This is intended to be used on signals (or images). Produces a PSNR value for
  each image in batch.

  The last three dimensions of input are expected to be [height, width, depth].

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute PSNR over tf.uint8 Tensors.
      psnr1 = tf.image.psnr(im1, im2, max_val=255)

      # Compute PSNR over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
      # psnr1 and psnr2 both have type tf.float32 and are almost equal.
  ```

  Arguments:
    a: First set of images.
    b: Second set of images.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    name: Namespace to embed the computation in.

  Returns:
    The scalar PSNR between a and b. The returned tensor has type `tf.float32`
    and shape [batch_size, 1].
  "
  [ a b max_val name ]
  (py/call-attr image "psnr"  a b max_val name ))

(defn random-brightness 
  "Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  Args:
    image: An image or images to adjust.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    The brightness-adjusted image(s).

  Raises:
    ValueError: if `max_delta` is negative.
  "
  [ image max_delta seed ]
  (py/call-attr image "random_brightness"  image max_delta seed ))

(defn random-contrast 
  "Adjust the contrast of an image or images by a random factor.

  Equivalent to `adjust_contrast()` but uses a `contrast_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: An image tensor with 3 or more dimensions.
    lower: float.  Lower bound for the random contrast factor.
    upper: float.  Upper bound for the random contrast factor.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    The contrast-adjusted image(s).

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  "
  [ image lower upper seed ]
  (py/call-attr image "random_contrast"  image lower upper seed ))

(defn random-crop 
  "Randomly crops a tensor to a given size.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed`
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  "
  [ value size seed name ]
  (py/call-attr image "random_crop"  value size seed name ))

(defn random-flip-left-right 
  "Randomly flip an image horizontally (left to right).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the
  second dimension, which is `width`.  Otherwise output the image as-is.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    A tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [ image seed ]
  (py/call-attr image "random_flip_left_right"  image seed ))

(defn random-flip-up-down 
  "Randomly flips an image vertically (upside down).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise output the image as-is.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    A tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [ image seed ]
  (py/call-attr image "random_flip_up_down"  image seed ))

(defn random-hue 
  "Adjust the hue of RGB images by a random factor.

  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta]`.

  `max_delta` must be in the interval `[0, 0.5]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    max_delta: float.  Maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `max_delta` is invalid.
  "
  [ image max_delta seed ]
  (py/call-attr image "random_hue"  image max_delta seed ))

(defn random-jpeg-quality 
  "Randomly changes jpeg encoding quality for inducing jpeg noise.

  `min_jpeg_quality` must be in the interval `[0, 100]` and less than
  `max_jpeg_quality`.
  `max_jpeg_quality` must be in the interval `[0, 100]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    min_jpeg_quality: Minimum jpeg encoding quality to use.
    max_jpeg_quality: Maximum jpeg encoding quality to use.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
  "
  [ image min_jpeg_quality max_jpeg_quality seed ]
  (py/call-attr image "random_jpeg_quality"  image min_jpeg_quality max_jpeg_quality seed ))

(defn random-saturation 
  "Adjust the saturation of RGB images by a random factor.

  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    lower: float.  Lower bound for the random saturation factor.
    upper: float.  Upper bound for the random saturation factor.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `upper <= lower` or if `lower < 0`.
  "
  [ image lower upper seed ]
  (py/call-attr image "random_saturation"  image lower upper seed ))

(defn resize 
  "Resize `images` to `size` using the specified `method`.

  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad`.

  When 'antialias' is true, the sampling filter will anti-alias the input image
  as well as interpolate.  When downsampling an image with [anti-aliasing](
  https://en.wikipedia.org/wiki/Spatial_anti-aliasing) the sampling filter
  kernel is scaled in order to properly anti-alias the input image signal.
  'antialias' has no effect when upsampling an image.

  *   <b>`bilinear`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation) If 'antialias' is
    true, becomes a hat/tent filter function with radius 1 when downsampling.
  *   <b>`lanczos3`</b>:  [Lanczos kernel](
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 3.
    High-quality practical filter but may have some ringing especially on
    synthetic images.
  *   <b>`lanczos5`</b>: [Lanczos kernel] (
    https://en.wikipedia.org/wiki/Lanczos_resampling) with radius 5.
    Very-high-quality filter but may have stronger ringing.
  *   <b>`bicubic`</b>: [Cubic interpolant](
    https://en.wikipedia.org/wiki/Bicubic_interpolation) of Keys. Equivalent to
    Catmull-Rom kernel. Reasonably good quality and faster than Lanczos3Kernel,
    particularly when upsampling.
  *   <b>`gaussian`</b>: [Gaussian kernel](
    https://en.wikipedia.org/wiki/Gaussian_filter) with radius 3,
    sigma = 1.5 / 3.0.
  *   <b>`nearest`</b>: [Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
    'antialias' has no effect when used with nearest neighbor interpolation.
  *   <b>`area`</b>: Anti-aliased resampling with area interpolation.
    'antialias' has no effect when used with area interpolation; it
    always anti-aliases.
  *   <b>`mitchellcubic`</b>: Mitchell-Netravali Cubic non-interpolating filter.
    For synthetic images (especially those lacking proper prefiltering), less
    ringing than Keys cubic kernel but less sharp.

  Note that near image edges the filtering kernel may be partially outside the
  image boundaries. For these pixels, only input pixels inside the image will be
  included in the filter sum, and the output value will be appropriately
  normalized.

  The return value has the same type as `images` if `method` is
  `ResizeMethod.NEAREST_NEIGHBOR`. Otherwise, the return value has type
  `float32`.

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: ResizeMethod.  Defaults to `bilinear`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
    name: A name for this operation (optional).

  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has invalid shape or type.
    ValueError: if an unsupported resize method is specified.

  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  "
  [images size & {:keys [method preserve_aspect_ratio antialias name]
                       :or {name None}} ]
    (py/call-attr-kw image "resize" [images size] {:method method :preserve_aspect_ratio preserve_aspect_ratio :antialias antialias :name name }))

(defn resize-with-crop-or-pad 
  "Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.

  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Cropped and/or padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  "
  [ image target_height target_width ]
  (py/call-attr image "resize_with_crop_or_pad"  image target_height target_width ))
(defn resize-with-pad 
  "Resizes and pads an image to a target width and height.

  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `image.resize()`
    antialias: Whether to use anti-aliasing when resizing. See 'image.resize()'.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  "
  [image target_height target_width  & {:keys [method antialias]} ]
    (py/call-attr-kw image "resize_with_pad" [image target_height target_width] {:method method :antialias antialias }))

(defn rgb-to-grayscale 
  "Converts one or more images from RGB to Grayscale.

  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.

  Args:
    images: The RGB tensor to convert. Last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).

  Returns:
    The converted grayscale image(s).
  "
  [ images name ]
  (py/call-attr image "rgb_to_grayscale"  images name ))

(defn rgb-to-hsv 
  "Converts one or more images from RGB to HSV.

  Outputs a tensor of the same shape as the `images` tensor, containing the HSV
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
  `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
  corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      1-D or higher rank. RGB data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  "
  [ images name ]
  (py/call-attr image "rgb_to_hsv"  images name ))

(defn rgb-to-yiq 
  "Converts one or more images from RGB to YIQ.

  Outputs a tensor of the same shape as the `images` tensor, containing the YIQ
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  "
  [ images ]
  (py/call-attr image "rgb_to_yiq"  images ))

(defn rgb-to-yuv 
  "Converts one or more images from RGB to YUV.

  Outputs a tensor of the same shape as the `images` tensor, containing the YUV
  value of the pixels.
  The output is only well defined if the value in images are in [0,1].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  "
  [ images ]
  (py/call-attr image "rgb_to_yuv"  images ))

(defn rot90 
  "Rotate image(s) counter-clockwise by 90 degrees.


  For example:
  ```python
  a=tf.constant([[[1],[2]],[[3],[4]]])
  # rotating `a` counter clockwise by 90 degrees
  a_rot=tf.image.rot90(a,k=1) #rotated `a`
  print(a_rot) # [[[2],[4]],[[1],[3]]]
  ```
  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name: A name for this operation (optional).

  Returns:
    A rotated tensor of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [image & {:keys [k name]
                       :or {name None}} ]
    (py/call-attr-kw image "rot90" [image] {:k k :name name }))

(defn sample-distorted-bounding-box 
  "Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_boxes` to
  visualize what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`.
  The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
  and height of the underlying image.

  For example,

  ```python
      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes,
          min_object_covered=0.1)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.compat.v1.summary.image('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)
  ```

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`. 1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`. 3-D with shape `[batch, N, 4]`
      describing the N bounding boxes associated with the image.
    seed: An optional `int`. Defaults to `0`. If `seed` is set to non-zero, the
      random number generator is seeded by the given `seed`.  Otherwise, it is
      seeded by a random seed.
    min_object_covered: A Tensor of type `float32`. Defaults to `0.1`. The
      cropped area of the image must contain at least this fraction of any
      bounding box supplied. The value of this parameter should be non-negative.
      In the case of 0, the cropped area does not need to overlap any of the
      bounding boxes supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75,
      1.33]`. The cropped area of the image must have an aspect `ratio = width /
      height` within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`. The
      cropped area of the image must contain a fraction of the supplied image
      within this range.
    max_attempts: An optional `int`. Defaults to `100`. Number of attempts at
      generating a cropped region of the image of the specified constraints.
      After `max_attempts` failures, return the entire image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied. If true, assume an
      implicit bounding box covering the whole input. If false, raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).

    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing
    `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing
    the distorted bounding box.
    Provide as input to `tf.image.draw_bounding_boxes`.
  "
  [image_size bounding_boxes & {:keys [seed min_object_covered aspect_ratio_range area_range max_attempts use_image_if_no_bounding_boxes name]
                       :or {aspect_ratio_range None area_range None max_attempts None use_image_if_no_bounding_boxes None name None}} ]
    (py/call-attr-kw image "sample_distorted_bounding_box" [image_size bounding_boxes] {:seed seed :min_object_covered min_object_covered :aspect_ratio_range aspect_ratio_range :area_range area_range :max_attempts max_attempts :use_image_if_no_bounding_boxes use_image_if_no_bounding_boxes :name name }))

(defn sobel-edges 
  "Returns a tensor holding Sobel edge maps.

  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
      float64.  The image(s) must be 2x2 or larger.

  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  "
  [ image ]
  (py/call-attr image "sobel_edges"  image ))
(defn ssim 
  "Computes SSIM index between img1 and img2.

  This function is based on the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If input is already YUV, then it will
  compute YUV SSIM average.)

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  The image sizes must be at least 11x11 because of the filter size.

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute SSIM over tf.uint8 Tensors.
      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)

      # Compute SSIM over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)
      # ssim1 and ssim2 both have type tf.float32 and are almost equal.
  ```

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we taken the values in range of 0< K2 <0.4).

  Returns:
    A tensor containing an SSIM value for each image in batch.  Returned SSIM
    values are in range (-1, 1], when pixel values are non-negative. Returns
    a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
  "
  [img1 img2 max_val  & {:keys [filter_size filter_sigma k1 k2]} ]
    (py/call-attr-kw image "ssim" [img1 img2 max_val] {:filter_size filter_size :filter_sigma filter_sigma :k1 k1 :k2 k2 }))
(defn ssim-multiscale 
  "Computes the MS-SSIM between img1 and img2.

  This function assumes that `img1` and `img2` are image batches, i.e. the last
  three dimensions are [height, width, channels].

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If input is already YUV, then it will
  compute YUV SSIM average.)

  Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. \"Multiscale
  structural similarity for image quality assessment.\" Signals, Systems and
  Computers, 2004.

  Arguments:
    img1: First image batch.
    img2: Second image batch. Must have the same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the original paper.
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we taken the values in range of 0< K2 <0.4).

  Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
  "
  [img1 img2 max_val  & {:keys [power_factors filter_size filter_sigma k1 k2]} ]
    (py/call-attr-kw image "ssim_multiscale" [img1 img2 max_val] {:power_factors power_factors :filter_size filter_size :filter_sigma filter_sigma :k1 k1 :k2 k2 }))

(defn total-variation 
  "Calculate and return the total variation for one or more images.

  The total variation is the sum of the absolute differences for neighboring
  pixel-values in the input images. This measures how much noise is in the
  images.

  This can be used as a loss-function during optimization so as to suppress
  noise in images. If you have a batch of images, then you should calculate
  the scalar loss-value as the sum:
  `loss = tf.reduce_sum(tf.image.total_variation(images))`

  This implements the anisotropic 2-D version of the formula described here:

  https://en.wikipedia.org/wiki/Total_variation_denoising

  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    name: A name for the operation (optional).

  Raises:
    ValueError: if images.shape is not a 3-D or 4-D vector.

  Returns:
    The total variation of `images`.

    If `images` was 4-D, return a 1-D float Tensor of shape `[batch]` with the
    total variation for each image in the batch.
    If `images` was 3-D, return a scalar float with the total variation for
    that image.
  "
  [ images name ]
  (py/call-attr image "total_variation"  images name ))

(defn transpose 
  "Transpose image(s) by swapping the height and width dimension.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    name: A name for this operation (optional).

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
   `[batch, width, height, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
   `[width, height, channels]`

  Raises:
    ValueError: if the shape of `image` not supported.
  "
  [ image name ]
  (py/call-attr image "transpose"  image name ))

(defn yiq-to-rgb 
  "Converts one or more images from YIQ to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  I value are in [-0.5957,0.5957] and Q value are in [-0.5226,0.5226].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  "
  [ images ]
  (py/call-attr image "yiq_to_rgb"  images ))

(defn yuv-to-rgb 
  "Converts one or more images from YUV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels.
  The output is only well defined if the Y value in images are in [0,1],
  U and V value are in [-0.5,0.5].

  Args:
    images: 2-D or higher rank. Image data to convert. Last dimension must be
      size 3.

  Returns:
    images: tensor with the same shape as `images`.
  "
  [ images ]
  (py/call-attr image "yuv_to_rgb"  images ))
