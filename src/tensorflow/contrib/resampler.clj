(ns tensorflow.contrib.resampler
  "Ops and modules related to resampler."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce resampler (import-module "tensorflow.contrib.resampler"))
(defn resampler 
  "Resamples input data at user defined coordinates.

  The resampler currently only supports bilinear interpolation of 2D data.

  Args:
    data: Tensor of shape `[batch_size, data_height, data_width,
      data_num_channels]` containing 2D data that will be resampled.
    warp: Tensor of minimum rank 2 containing the coordinates at which
      resampling will be performed. Since only bilinear interpolation is
      currently supported, the last dimension of the `warp` tensor must be 2,
      representing the (x, y) coordinate where x is the index for width and y is
      the index for height.
    name: Optional name of the op.

  Returns:
    Tensor of resampled values from `data`. The output tensor shape is
    determined by the shape of the warp tensor. For example, if `data` is of
    shape `[batch_size, data_height, data_width, data_num_channels]` and warp of
    shape `[batch_size, dim_0, ... , dim_n, 2]` the output will be of shape
    `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

  Raises:
    ImportError: if the wrapper generated during compilation is not present when
    the function is called.
  "
  [data warp  & {:keys [name]} ]
    (py/call-attr-kw resampler "resampler" [data warp] {:name name }))
