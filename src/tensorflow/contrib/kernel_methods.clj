(ns tensorflow.contrib.kernel-methods
  "Ops and estimators that enable explicit kernel methods in TensorFlow.

@@KernelLinearClassifier
@@RandomFourierFeatureMapper
@@sparse_multiclass_hinge_loss
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce kernel-methods (import-module "tensorflow.contrib.kernel_methods"))

(defn sparse-multiclass-hinge-loss 
  "Adds Ops for computing the multiclass hinge loss.

  The implementation is based on the following paper:
  On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
  by Crammer and Singer.
  link: http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf

  This is a generalization of standard (binary) hinge loss. For a given instance
  with correct label c*, the loss is given by:
    $$loss = max_{c != c*} logits_c - logits_{c*} + 1.$$
  or equivalently
    $$loss = max_c { logits_c - logits_{c*} + I_{c != c*} }$$
  where \\(I_{c != c*} = 1\ \text{if}\ c != c*\\) and 0 otherwise.

  Args:
    labels: `Tensor` of shape [batch_size] or [batch_size, 1]. Corresponds to
      the ground truth. Each entry must be an index in `[0, num_classes)`.
    logits: `Tensor` of shape [batch_size, num_classes] corresponding to the
      unscaled logits. Its dtype should be either `float32` or `float64`.
    weights: Optional (python) scalar or `Tensor`. If a non-scalar `Tensor`, its
      rank should be either 1 ([batch_size]) or 2 ([batch_size, 1]).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is a scalar.

  Raises:
    ValueError: If `logits`, `labels` or `weights` have invalid or inconsistent
      shapes.
    ValueError: If `labels` tensor has invalid dtype.
  "
  [labels logits & {:keys [weights scope loss_collection reduction]
                       :or {scope None}} ]
    (py/call-attr-kw kernel-methods "sparse_multiclass_hinge_loss" [labels logits] {:weights weights :scope scope :loss_collection loss_collection :reduction reduction }))
