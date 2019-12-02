(ns tensorflow.contrib.kernel-methods.RandomFourierFeatureMapper
  "Class that implements Random Fourier Feature Mapping (RFFM) in TensorFlow.

  The RFFM mapping is used to approximate the Gaussian (RBF) kernel:
  $$(exp(-||x-y||_2^2 / (2 * \sigma^2))$$

  The implementation of RFFM is based on the following paper:
  \"Random Features for Large-Scale Kernel Machines\" by Ali Rahimi and Ben Recht.
  (link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

  The mapping uses a matrix \\(\Omega \in R^{d x D}\\) and a bias vector
  \\(b \in R^D\\) where \\(d\\) is the input dimension (number of dense input
  features) and \\(D\\) is the output dimension (i.e., dimension of the feature
  space the input is mapped to). Each entry of \\(\Omega\\) is sampled i.i.d.
  from a (scaled) Gaussian distribution and each entry of \\(b\\) is sampled
  independently and uniformly from [0, \\(2 * \pi\\)].

  For a single input feature vector \\(x \in R^d\\), its RFFM is defined as:
  $$\sqrt(2/D) * cos(x * \Omega + b)$$

  where \\(cos\\) is the element-wise cosine function and \\(x, b\\) are
  represented as row vectors. The aforementioned paper shows that the linear
  kernel of RFFM-mapped vectors approximates the Gaussian kernel of the initial
  vectors.

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

(defn RandomFourierFeatureMapper 
  "Class that implements Random Fourier Feature Mapping (RFFM) in TensorFlow.

  The RFFM mapping is used to approximate the Gaussian (RBF) kernel:
  $$(exp(-||x-y||_2^2 / (2 * \sigma^2))$$

  The implementation of RFFM is based on the following paper:
  \"Random Features for Large-Scale Kernel Machines\" by Ali Rahimi and Ben Recht.
  (link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

  The mapping uses a matrix \\(\Omega \in R^{d x D}\\) and a bias vector
  \\(b \in R^D\\) where \\(d\\) is the input dimension (number of dense input
  features) and \\(D\\) is the output dimension (i.e., dimension of the feature
  space the input is mapped to). Each entry of \\(\Omega\\) is sampled i.i.d.
  from a (scaled) Gaussian distribution and each entry of \\(b\\) is sampled
  independently and uniformly from [0, \\(2 * \pi\\)].

  For a single input feature vector \\(x \in R^d\\), its RFFM is defined as:
  $$\sqrt(2/D) * cos(x * \Omega + b)$$

  where \\(cos\\) is the element-wise cosine function and \\(x, b\\) are
  represented as row vectors. The aforementioned paper shows that the linear
  kernel of RFFM-mapped vectors approximates the Gaussian kernel of the initial
  vectors.

  "
  [input_dim output_dim & {:keys [stddev seed name]
                       :or {name None}} ]
    (py/call-attr-kw kernel-methods "RandomFourierFeatureMapper" [input_dim output_dim] {:stddev stddev :seed seed :name name }))

(defn input-dim 
  ""
  [ self ]
    (py/call-attr self "input_dim"))

(defn map 
  "Maps each row of input_tensor using random Fourier features.

    Args:
      input_tensor: a `Tensor` containing input features. It's shape is
      [batch_size, self._input_dim].

    Returns:
      A `Tensor` of shape [batch_size, self._output_dim] containing RFFM-mapped
      features.

    Raises:
      InvalidShapeError: if the shape of the `input_tensor` is inconsistent with
        expected input dimension.
    "
  [ self input_tensor ]
  (py/call-attr self "map"  self input_tensor ))

(defn name 
  "Returns a name for the `RandomFourierFeatureMapper` instance.

    If the name provided in the constructor is `None`, then the object's unique
    id is returned.

    Returns:
      A name for the `RandomFourierFeatureMapper` instance.
    "
  [ self ]
    (py/call-attr self "name"))

(defn output-dim 
  ""
  [ self ]
    (py/call-attr self "output_dim"))
