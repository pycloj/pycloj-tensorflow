(ns tensorflow.contrib.keras.api.keras.preprocessing.image
  "Keras data preprocessing utils for image data."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow.contrib.keras.api.keras.preprocessing.image"))

(defn array-to-img 
  "Converts a 3D Numpy array to a PIL Image instance.

  Arguments:
      x: Input Numpy array.
      data_format: Image data format.
          either \"channels_first\" or \"channels_last\".
      scale: Whether to rescale image values
          to be within `[0, 255]`.
      dtype: Dtype to use.

  Returns:
      A PIL Image instance.

  Raises:
      ImportError: if PIL is not available.
      ValueError: if invalid `x` or `data_format` is passed.
  "
  [x data_format & {:keys [scale dtype]
                       :or {dtype None}} ]
    (py/call-attr-kw image "array_to_img" [x data_format] {:scale scale :dtype dtype }))

(defn img-to-array 
  "Converts a PIL Image instance to a Numpy array.

  Arguments:
      img: PIL Image instance.
      data_format: Image data format,
          either \"channels_first\" or \"channels_last\".
      dtype: Dtype to use for the returned array.

  Returns:
      A 3D Numpy array.

  Raises:
      ValueError: if invalid `img` or `data_format` is passed.
  "
  [ img data_format dtype ]
  (py/call-attr image "img_to_array"  img data_format dtype ))

(defn load-img 
  "Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode=\"grayscale\"`.
        color_mode: One of \"grayscale\", \"rgb\", \"rgba\". Default: \"rgb\".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are \"nearest\", \"bilinear\", and \"bicubic\".
            If PIL version 1.1.3 or newer is installed, \"lanczos\" is also
            supported. If PIL version 3.4.0 or newer is installed, \"box\" and
            \"hamming\" are also supported. By default, \"nearest\" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    "
  [path & {:keys [grayscale color_mode target_size interpolation]
                       :or {target_size None}} ]
    (py/call-attr-kw image "load_img" [path] {:grayscale grayscale :color_mode color_mode :target_size target_size :interpolation interpolation }))
(defn random-channel-shift 
  "Performs a random channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.
    "
  [x intensity_range  & {:keys [channel_axis]} ]
    (py/call-attr-kw image "random_channel_shift" [x intensity_range] {:channel_axis channel_axis }))
(defn random-rotation 
  "Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Rotated Numpy image tensor.
    "
  [x rg  & {:keys [row_axis col_axis channel_axis fill_mode cval interpolation_order]} ]
    (py/call-attr-kw image "random_rotation" [x rg] {:row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))
(defn random-shear 
  "Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Sheared Numpy image tensor.
    "
  [x intensity  & {:keys [row_axis col_axis channel_axis fill_mode cval interpolation_order]} ]
    (py/call-attr-kw image "random_shear" [x intensity] {:row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))
(defn random-shift 
  "Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Shifted Numpy image tensor.
    "
  [x wrg hrg  & {:keys [row_axis col_axis channel_axis fill_mode cval interpolation_order]} ]
    (py/call-attr-kw image "random_shift" [x wrg hrg] {:row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))
(defn random-zoom 
  "Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    "
  [x zoom_range  & {:keys [row_axis col_axis channel_axis fill_mode cval interpolation_order]} ]
    (py/call-attr-kw image "random_zoom" [x zoom_range] {:row_axis row_axis :col_axis col_axis :channel_axis channel_axis :fill_mode fill_mode :cval cval :interpolation_order interpolation_order }))
