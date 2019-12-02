(ns tensorflow.contrib.linear-optimizer.SparseFeatureColumn
  "Represents a sparse feature column.

  Contains three tensors representing a sparse feature column, they are
  example indices (`int64`), feature indices (`int64`), and feature
  values (`float`).
  Feature weights are optional, and are treated as `1.0f` if missing.

  For example, consider a batch of 4 examples, which contains the following
  features in a particular `SparseFeatureColumn`:

  * Example 0: feature 5, value 1
  * Example 1: feature 6, value 1 and feature 10, value 0.5
  * Example 2: no features
  * Example 3: two copies of feature 2, value 1

  This SparseFeatureColumn will be represented as follows:

  ```
   <0, 5,  1>
   <1, 6,  1>
   <1, 10, 0.5>
   <3, 2,  1>
   <3, 2,  1>
  ```

  For a batch of 2 examples below:

  * Example 0: feature 5
  * Example 1: feature 6

  is represented by `SparseFeatureColumn` as:

  ```
   <0, 5,  1>
   <1, 6,  1>

  ```

  @@__init__
  @@example_indices
  @@feature_indices
  @@feature_values
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce linear-optimizer (import-module "tensorflow.contrib.linear_optimizer"))

(defn SparseFeatureColumn 
  "Represents a sparse feature column.

  Contains three tensors representing a sparse feature column, they are
  example indices (`int64`), feature indices (`int64`), and feature
  values (`float`).
  Feature weights are optional, and are treated as `1.0f` if missing.

  For example, consider a batch of 4 examples, which contains the following
  features in a particular `SparseFeatureColumn`:

  * Example 0: feature 5, value 1
  * Example 1: feature 6, value 1 and feature 10, value 0.5
  * Example 2: no features
  * Example 3: two copies of feature 2, value 1

  This SparseFeatureColumn will be represented as follows:

  ```
   <0, 5,  1>
   <1, 6,  1>
   <1, 10, 0.5>
   <3, 2,  1>
   <3, 2,  1>
  ```

  For a batch of 2 examples below:

  * Example 0: feature 5
  * Example 1: feature 6

  is represented by `SparseFeatureColumn` as:

  ```
   <0, 5,  1>
   <1, 6,  1>

  ```

  @@__init__
  @@example_indices
  @@feature_indices
  @@feature_values
  "
  [ example_indices feature_indices feature_values ]
  (py/call-attr linear-optimizer "SparseFeatureColumn"  example_indices feature_indices feature_values ))

(defn example-indices 
  "The example indices represented as a dense tensor.

    Returns:
      A 1-D Tensor of int64 with shape `[N]`.
    "
  [ self ]
    (py/call-attr self "example_indices"))

(defn feature-indices 
  "The feature indices represented as a dense tensor.

    Returns:
      A 1-D Tensor of int64 with shape `[N]`.
    "
  [ self ]
    (py/call-attr self "feature_indices"))

(defn feature-values 
  "The feature values represented as a dense tensor.

    Returns:
      May return None, or a 1-D Tensor of float32 with shape `[N]`.
    "
  [ self ]
    (py/call-attr self "feature_values"))
