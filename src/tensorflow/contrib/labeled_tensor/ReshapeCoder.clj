(ns tensorflow.contrib.labeled-tensor.ReshapeCoder
  "Utility class for mapping to and from another shape.

  For example, say you have a function `crop_center` which expects a
  LabeledTensor with axes named ['batch', 'row', 'column', 'depth'], and
  you have a LabeledTensor `masked_image_lt` with axes ['batch', 'row',
  'column', 'channel', 'mask'].

  To call `crop_center` with `masked_image_lt` you'd normally have to write:

  >>> reshape_lt = lt.reshape(masked_image_lt, ['channel', 'mask'], ['depth'])
  >>> crop_lt = crop_center(reshape_lt)
  >>> result_lt = lt.reshape(crop_lt, ['depth'],
  ...   [masked_image_lt.axes['channel'], masked_image_lt.axes['mask']])

  ReshapeCoder takes care of this renaming logic for you, allowing you to
  instead write:

  >>> rc = ReshapeCoder(['channel', 'mask'], ['depth'])
  >>> result_lt = rc.decode(crop_center(rc.encode(masked_image_lt)))

  Here, `decode` restores the original axes 'channel' and 'mask', so
  `crop_center` must not have modified the size of the 'depth' axis.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce labeled-tensor (import-module "tensorflow.contrib.labeled_tensor"))

(defn ReshapeCoder 
  "Utility class for mapping to and from another shape.

  For example, say you have a function `crop_center` which expects a
  LabeledTensor with axes named ['batch', 'row', 'column', 'depth'], and
  you have a LabeledTensor `masked_image_lt` with axes ['batch', 'row',
  'column', 'channel', 'mask'].

  To call `crop_center` with `masked_image_lt` you'd normally have to write:

  >>> reshape_lt = lt.reshape(masked_image_lt, ['channel', 'mask'], ['depth'])
  >>> crop_lt = crop_center(reshape_lt)
  >>> result_lt = lt.reshape(crop_lt, ['depth'],
  ...   [masked_image_lt.axes['channel'], masked_image_lt.axes['mask']])

  ReshapeCoder takes care of this renaming logic for you, allowing you to
  instead write:

  >>> rc = ReshapeCoder(['channel', 'mask'], ['depth'])
  >>> result_lt = rc.decode(crop_center(rc.encode(masked_image_lt)))

  Here, `decode` restores the original axes 'channel' and 'mask', so
  `crop_center` must not have modified the size of the 'depth' axis.
  "
  [ existing_axis_names new_axes name ]
  (py/call-attr labeled-tensor "ReshapeCoder"  existing_axis_names new_axes name ))

(defn decode 
  "Reshape the input to the original shape.

    This is the inverse of encode.
    Encode must have been called at least once prior to this method being
    called.

    Args:
      labeled_tensor: The input tensor.

    Returns:
      The input reshaped to the original shape.

    Raises:
      ValueError: If this method was called before encode was called.
    "
  [ self labeled_tensor ]
  (py/call-attr self "decode"  self labeled_tensor ))

(defn encode 
  "Reshape the input to the target shape.

    If called several times, the axes named in existing_axis_names must be
    identical.

    Args:
      labeled_tensor: The input tensor.

    Returns:
      The input reshaped to the target shape.

    Raises:
      ValueError: If the axes in existing_axis_names don't match the axes of
        a tensor in a previous invocation of this method.
    "
  [ self labeled_tensor ]
  (py/call-attr self "encode"  self labeled_tensor ))
