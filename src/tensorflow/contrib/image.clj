(ns tensorflow.contrib.image
  "Ops for image manipulation.

### API

This module provides functions for image manipulation; currently, chrominance
transforms (including changing saturation and hue) in YIQ space and
projective transforms (including rotation) are supported.

## Image Transformation `Ops`

@@angles_to_projective_transforms
@@compose_transforms
@@adjust_yiq_hsv
@@flat_transforms_to_matrices
@@matrices_to_flat_transforms
@@random_yiq_hsv
@@rotate
@@transform
@@translate
@@translations_to_projective_transforms
@@dense_image_warp
@@interpolate_spline
@@sparse_image_warp

## Image Segmentation `Ops`

@@connected_components

## Matching `Ops`

@@bipartite_match

## Random Dot Stereogram `Ops`

@@single_image_random_dot_stereograms
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce image (import-module "tensorflow.contrib.image"))

(defn angles-to-projective-transforms 
  "Returns projective transform(s) for the given angle(s).

  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`.
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to `tf.contrib.image.transform`.
  "
  [ angles image_height image_width name ]
  (py/call-attr image "angles_to_projective_transforms"  angles image_height image_width name ))
(defn bipartite-match 
  "Find bipartite matching based on a given distance matrix.

  A greedy bi-partite matching algorithm is used to obtain the matching with
  the (greedy) minimum distance.

  Args:
    distance_mat: A 2-D float tensor of shape `[num_rows, num_columns]`. It is a
      pair-wise distance matrix between the entities represented by each row and
      each column. It is an asymmetric matrix. The smaller the distance is, the
      more similar the pairs are. The bipartite matching is to minimize the
      distances.
    num_valid_rows: A scalar or a 1-D tensor with one element describing the
      number of valid rows of distance_mat to consider for the bipartite
      matching. If set to be negative, then all rows from `distance_mat` are
      used.
    top_k: A scalar that specifies the number of top-k matches to retrieve.
      If set to be negative, then is set according to the maximum number of
      matches from `distance_mat`.
    name: The name of the op.

  Returns:
    row_to_col_match_indices: A vector of length num_rows, which is the number
      of rows of the input `distance_matrix`. If `row_to_col_match_indices[i]`
      is not -1, row i is matched to column `row_to_col_match_indices[i]`.
    col_to_row_match_indices: A vector of length num_columns, which is the
      number of columns of the input distance matrix.
      If `col_to_row_match_indices[j]` is not -1, column j is matched to row
      `col_to_row_match_indices[j]`.
  "
  [distance_mat num_valid_rows  & {:keys [top_k name]} ]
    (py/call-attr-kw image "bipartite_match" [distance_mat num_valid_rows] {:top_k top_k :name name }))

(defn compose-transforms 
  "Composes the transforms tensors.

  Args:
    *transforms: List of image projective transforms to be composed. Each
        transform is length 8 (single transform) or shape (N, 8) (batched
        transforms). The shapes of all inputs must be equal, and at least one
        input must be given.

  Returns:
    A composed transform tensor. When passed to `tf.contrib.image.transform`,
        equivalent to applying each of the given transforms to the image in
        order.
  "
  [  ]
  (py/call-attr image "compose_transforms"  ))

(defn connected-components 
  "Labels the connected components in a batch of images.

  A component is a set of pixels in a single input image, which are all adjacent
  and all have the same non-zero value. The components using a squared
  connectivity of one (all True entries are joined with their neighbors above,
  below, left, and right). Components across all images have consecutive ids 1
  through n. Components are labeled according to the first pixel of the
  component appearing in row-major order (lexicographic order by
  image_index_in_batch, row, col). Zero entries all have an output id of 0.

  This op is equivalent with `scipy.ndimage.measurements.label` on a 2D array
  with the default structuring element (which is the connectivity used here).

  Args:
    images: A 2D (H, W) or 3D (N, H, W) Tensor of boolean image(s).

  Returns:
    Components with the same shape as `images`. False entries in `images` have
    value 0, and all True entries map to a component id > 0.

  Raises:
    TypeError: if `images` is not 2D or 3D.
  "
  [ images ]
  (py/call-attr image "connected_components"  images ))
(defn dense-image-warp 
  "Image warping using per-pixel flow vectors.

  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by bilinear
  interpolation of the 4 nearest pixels around
  (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.


  Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).

    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.

  Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.

  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  "
  [image flow  & {:keys [name]} ]
    (py/call-attr-kw image "dense_image_warp" [image flow] {:name name }))

(defn flat-transforms-to-matrices 
  "Converts `tf.contrib.image` projective transforms to affine matrices.

  Note that the output matrices map output coordinates to input coordinates. For
  the forward transformation matrix, call `tf.linalg.inv` on the result.

  Args:
    transforms: Vector of length 8, or batches of transforms with shape
      `(N, 8)`.

  Returns:
    3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
      *output coordinates* (in homogeneous coordinates) of each transform to the
      corresponding *input coordinates*.

  Raises:
    ValueError: If `transforms` have an invalid shape.
  "
  [ transforms ]
  (py/call-attr image "flat_transforms_to_matrices"  transforms ))
(defn interpolate-spline 
  "Interpolate signal using polyharmonic interpolation.

  The interpolant has the form
  $$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$

  This is a sum of two terms: (1) a weighted sum of radial basis function (RBF)
  terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term with a bias.
  The \\(c_i\\) vectors are 'training' points. In the code, b is absorbed into v
  by appending 1 as a final dimension to x. The coefficients w and v are
  estimated such that the interpolant exactly fits the value of the function at
  the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\), and the
  vector w sums to 0. With these constraints, the coefficients can be obtained
  by solving a linear system.

  \\(\phi\\) is an RBF, parametrized by an interpolation
  order. Using order=2 produces the well-known thin-plate spline.

  We also provide the option to perform regularized interpolation. Here, the
  interpolant is selected to trade off between the squared loss on the training
  data and a certain measure of its curvature
  ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
  Using a regularization weight greater than zero has the effect that the
  interpolant will no longer exactly fit the training data. However, it may be
  less vulnerable to overfitting, particularly for high-order interpolation.

  Note the interpolation procedure is differentiable with respect to all inputs
  besides the order parameter.

  We support dynamically-shaped inputs, where batch_size, n, and m are None
  at graph construction time. However, d and k must be known.

  Args:
    train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
      locations. These do not need to be regularly-spaced.
    train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional values
      evaluated at train_points.
    query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
      where we will output the interpolant's values.
    order: order of the interpolation. Common values are 1 for
      \\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\) (thin-plate spline),
       or 3 for \\(\phi(r) = r^3\\).
    regularization_weight: weight placed on the regularization term.
      This will depend substantially on the problem, and it should always be
      tuned. For many problems, it is reasonable to use no regularization.
      If using a non-zero value, we recommend a small value like 0.001.
    name: name prefix for ops created by this function

  Returns:
    `[b, m, k]` float `Tensor` of query values. We use train_points and
    train_values to perform polyharmonic interpolation. The query values are
    the values of the interpolant evaluated at the locations specified in
    query_points.
  "
  [train_points train_values query_points order  & {:keys [regularization_weight name]} ]
    (py/call-attr-kw image "interpolate_spline" [train_points train_values query_points order] {:regularization_weight regularization_weight :name name }))

(defn matrices-to-flat-transforms 
  "Converts affine matrices to `tf.contrib.image` projective transforms.

  Note that we expect matrices that map output coordinates to input coordinates.
  To convert forward transformation matrices, call `tf.linalg.inv` on the
  matrices and use the result here.

  Args:
    transform_matrices: One or more affine transformation matrices, for the
      reverse transformation in homogeneous coordinates. Shape `(3, 3)` or
      `(N, 3, 3)`.

  Returns:
    2D tensor of flat transforms with shape `(N, 8)`, which may be passed into
      `tf.contrib.image.transform`.

  Raises:
    ValueError: If `transform_matrices` have an invalid shape.
  "
  [ transform_matrices ]
  (py/call-attr image "matrices_to_flat_transforms"  transform_matrices ))

(defn rotate 
  "Rotate image(s) counterclockwise by the passed angle(s) in radians.

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW). The rank must be statically known (the
       shape is not `TensorShape(None)`.
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.
    interpolation: Interpolation mode. Supported values: \"NEAREST\", \"BILINEAR\".
    name: The name of the op.

  Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  "
  [images angles & {:keys [interpolation name]
                       :or {name None}} ]
    (py/call-attr-kw image "rotate" [images angles] {:interpolation interpolation :name name }))

(defn single-image-random-dot-stereograms 
  "Output a RandomDotStereogram Tensor for export via encode_PNG/JPG OP.

  Given the 2-D tensor 'depth_values' with encoded Z values, this operation
  will encode 3-D data into a 2-D image.  The output of this Op is suitable
  for the encode_PNG/JPG ops.  Be careful with image compression as this may
  corrupt the encode 3-D data within the image.

  Based upon [this
  paper](https://www.cs.waikato.ac.nz/~ihw/papers/94-HWT-SI-IHW-SIRDS-paper.pdf).

  This outputs a SIRDS image as picture_out.png:

  ```python
  img=[[1,2,3,3,2,1],
       [1,2,3,4,5,2],
       [1,2,3,4,5,3],
       [1,2,3,4,5,4],
       [6,5,4,4,5,5]]
  session = tf.compat.v1.InteractiveSession()
  sirds = single_image_random_dot_stereograms(
      img,
      convergence_dots_size=8,
      number_colors=256,normalize=True)

  out = sirds.eval()
  png = tf.image.encode_png(out).eval()
  with open('picture_out.png', 'wb') as f:
    f.write(png)
  ```

  Args:
    depth_values: A `Tensor`. Must be one of the following types:
      `float64`, `float32`, `int64`, `int32`.  Z values of data to encode
      into 'output_data_window' window, lower further away {0.0 floor(far),
      1.0 ceiling(near) after norm}, must be 2-D tensor
    hidden_surface_removal: An optional `bool`. Defaults to `True`.
      Activate hidden surface removal
    convergence_dots_size: An optional `int`. Defaults to `8`.
      Black dot size in pixels to help view converge image, drawn on bottom
      of the image
    dots_per_inch: An optional `int`. Defaults to `72`.
      Output device in dots/inch
    eye_separation: An optional `float`. Defaults to `2.5`.
      Separation between eyes in inches
    mu: An optional `float`. Defaults to `0.3333`.
      Depth of field, Fraction of viewing distance (eg. 1/3 = 0.3333)
    normalize: An optional `bool`. Defaults to `True`.
      Normalize input data to [0.0, 1.0]
    normalize_max: An optional `float`. Defaults to `-100`.
      Fix MAX value for Normalization (0.0) - if < MIN, autoscale
    normalize_min: An optional `float`. Defaults to `100`.
      Fix MIN value for Normalization (0.0) - if > MAX, autoscale
    border_level: An optional `float`. Defaults to `0`.
      Value of bord in depth 0.0 {far} to 1.0 {near}
    number_colors: An optional `int`. Defaults to `256`. 2 (Black &
      White), 256 (grayscale), and Numbers > 256 (Full Color) are
      supported
    output_image_shape: An optional `tf.TensorShape` or list of `ints`.
      Defaults to shape `[1024, 768, 1]`. Defines output shape of returned
      image in '[X,Y, Channels]' 1-grayscale, 3 color; channels will be
      updated to 3 if number_colors > 256
    output_data_window: An optional `tf.TensorShape` or list of `ints`.
      Defaults to `[1022, 757]`. Size of \"DATA\" window, must be equal to or
      smaller than `output_image_shape`, will be centered and use
      `convergence_dots_size` for best fit to avoid overlap if possible

  Returns:
    A `Tensor` of type `uint8` of shape 'output_image_shape' with encoded
    'depth_values'
  "
  [ depth_values hidden_surface_removal convergence_dots_size dots_per_inch eye_separation mu normalize normalize_max normalize_min border_level number_colors output_image_shape output_data_window ]
  (py/call-attr image "single_image_random_dot_stereograms"  depth_values hidden_surface_removal convergence_dots_size dots_per_inch eye_separation mu normalize normalize_max normalize_min border_level number_colors output_image_shape output_data_window ))
(defn sparse-image-warp 
  "Image warping using correspondences between sparse control points.

  Apply a non-linear warp to the image, where the warp is specified by
  the source and destination locations of a (potentially small) number of
  control points. First, we use a polyharmonic spline
  (`tf.contrib.image.interpolate_spline`) to interpolate the displacements
  between the corresponding control points to a dense flow field.
  Then, we warp the image using this dense flow field
  (`tf.contrib.image.dense_image_warp`).

  Let t index our control points. For regularization_weight=0, we have:
  warped_image[b, dest_control_point_locations[b, t, 0],
                  dest_control_point_locations[b, t, 1], :] =
  image[b, source_control_point_locations[b, t, 0],
           source_control_point_locations[b, t, 1], :].

  For regularization_weight > 0, this condition is met approximately, since
  regularized interpolation trades off smoothness of the interpolant vs.
  reconstruction of the interpolant at the control points.
  See `tf.contrib.image.interpolate_spline` for further documentation of the
  interpolation_order and regularization_weight arguments.


  Args:
    image: `[batch, height, width, channels]` float `Tensor`
    source_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    dest_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    interpolation_order: polynomial order used by the spline interpolation
    regularization_weight: weight on smoothness regularizer in interpolation
    num_boundary_points: How many zero-flow boundary points to include at
      each image edge.Usage:
        num_boundary_points=0: don't add zero-flow points
        num_boundary_points=1: 4 corners of the image
        num_boundary_points=2: 4 corners and one in the middle of each edge
          (8 points total)
        num_boundary_points=n: 4 corners and n-1 along each edge
    name: A name for the operation (optional).

    Note that image and offsets can be of type tf.half, tf.float32, or
    tf.float64, and do not necessarily have to be the same type.

  Returns:
    warped_image: `[batch, height, width, channels]` float `Tensor` with same
      type as input image.
    flow_field: `[batch, height, width, 2]` float `Tensor` containing the dense
      flow field produced by the interpolation.
  "
  [image source_control_point_locations dest_control_point_locations  & {:keys [interpolation_order regularization_weight num_boundary_points name]} ]
    (py/call-attr-kw image "sparse_image_warp" [image source_control_point_locations dest_control_point_locations] {:interpolation_order interpolation_order :regularization_weight regularization_weight :num_boundary_points num_boundary_points :name name }))

(defn transform 
  "Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW). The rank must be statically known (the
       shape is not `TensorShape(None)`.
    transforms: Projective transform matrix/matrices. A vector of length 8 or
       tensor of size N x 8. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
       the transform mapping input points to output points. Note that gradients
       are not backpropagated into transformation parameters.
    interpolation: Interpolation mode. Supported values: \"NEAREST\", \"BILINEAR\".
    output_shape: Output dimesion after the transform, [height, width].
       If None, output is the same size as input image.

    name: The name of the op.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
    ValueError: If output shape is not 1-D int32 Tensor.
  "
  [images transforms & {:keys [interpolation output_shape name]
                       :or {output_shape None name None}} ]
    (py/call-attr-kw image "transform" [images transforms] {:interpolation interpolation :output_shape output_shape :name name }))

(defn translate 
  "Translate image(s) by the passed vectors(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
        (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW). The rank must be statically known (the
        shape is not `TensorShape(None)`.
    translations: A vector representing [dx, dy] or (if images has rank 4)
        a matrix of length num_images, with a [dx, dy] vector for each image in
        the batch.
    interpolation: Interpolation mode. Supported values: \"NEAREST\", \"BILINEAR\".
    name: The name of the op.

  Returns:
    Image(s) with the same type and shape as `images`, translated by the given
        vector(s). Empty space due to the translation will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
  "
  [images translations & {:keys [interpolation name]
                       :or {name None}} ]
    (py/call-attr-kw image "translate" [images translations] {:interpolation interpolation :name name }))

(defn translations-to-projective-transforms 
  "Returns projective transform(s) for the given translation(s).

  Args:
      translations: A 2-element list representing [dx, dy] or a matrix of
          2-element lists representing [dx, dy] to translate for each image
          (for a batch of images). The rank must be statically known (the shape
          is not `TensorShape(None)`.
      name: The name of the op.

  Returns:
      A tensor of shape (num_images, 8) projective transforms which can be given
          to `tf.contrib.image.transform`.
  "
  [ translations name ]
  (py/call-attr image "translations_to_projective_transforms"  translations name ))
