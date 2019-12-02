(ns tensorflow.contrib.keras.api.keras.activations
  "Keras built-in activation functions."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce activations (import-module "tensorflow.contrib.keras.api.keras.activations"))

(defn deserialize 
  ""
  [ name custom_objects ]
  (py/call-attr activations "deserialize"  name custom_objects ))
(defn elu 
  "Exponential linear unit.

  Arguments:
      x: Input tensor.
      alpha: A scalar, slope of negative section.

  Returns:
      The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

  Reference:
      - [Fast and Accurate Deep Network Learning by Exponential
        Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
  "
  [x  & {:keys [alpha]} ]
    (py/call-attr-kw activations "elu" [x] {:alpha alpha }))

(defn get 
  ""
  [ identifier ]
  (py/call-attr activations "get"  identifier ))

(defn hard-sigmoid 
  "Hard sigmoid activation function.

  Faster to compute than sigmoid activation.

  Arguments:
      x: Input tensor.

  Returns:
      Hard sigmoid activation:
      - `0` if `x < -2.5`
      - `1` if `x > 2.5`
      - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
  "
  [ x ]
  (py/call-attr activations "hard_sigmoid"  x ))

(defn linear 
  "Linear activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The linear activation: `x`.
  "
  [ x ]
  (py/call-attr activations "linear"  x ))

(defn relu 
  "Rectified Linear Unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  "
  [x & {:keys [alpha max_value threshold]
                       :or {max_value None}} ]
    (py/call-attr-kw activations "relu" [x] {:alpha alpha :max_value max_value :threshold threshold }))

(defn selu 
  "Scaled Exponential Linear Unit (SELU).

  The Scaled Exponential Linear Unit (SELU) activation function is:
  `scale * x` if `x > 0` and `scale * alpha * (exp(x) - 1)` if `x < 0`
  where `alpha` and `scale` are pre-defined constants
  (`alpha = 1.67326324`
  and `scale = 1.05070098`).
  The SELU activation function multiplies  `scale` > 1 with the
  `[elu](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/elu)`
  (Exponential Linear Unit (ELU)) to ensure a slope larger than one
  for positive net inputs.

  The values of `alpha` and `scale` are
  chosen so that the mean and variance of the inputs are preserved
  between two consecutive layers as long as the weights are initialized
  correctly (see [`lecun_normal` initialization]
  (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal))
  and the number of inputs is \"large enough\"
  (see references for more information).

  ![](https://cdn-images-1.medium.com/max/1600/1*m0e8lZU_Zrkh4ESfQkY2Pw.png)
  (Courtesy: Blog on Towards DataScience at
  https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9)

  Example Usage:
  ```python3
  n_classes = 10 #10-class problem
  model = models.Sequential()
  model.add(Dense(64, kernel_initializer='lecun_normal', activation='selu',
  input_shape=(28, 28, 1))))
  model.add(Dense(32, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(n_classes, activation='softmax'))
  ```

  Arguments:
      x: A tensor or variable to compute the activation function for.

  Returns:
      The scaled exponential unit activation: `scale * elu(x, alpha)`.

  # Note
      - To be used together with the initialization \"[lecun_normal]
      (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal)\".
      - To be used together with the dropout variant \"[AlphaDropout]
      (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout)\".

  References:
      [Self-Normalizing Neural Networks (Klambauer et al, 2017)]
      (https://arxiv.org/abs/1706.02515)
  "
  [ x ]
  (py/call-attr activations "selu"  x ))

(defn serialize 
  ""
  [ activation ]
  (py/call-attr activations "serialize"  activation ))

(defn sigmoid 
  "Sigmoid.

  Applies the sigmoid activation function. The sigmoid function is defined as
  1 divided by (1 + exp(-x)). It's curve is like an \"S\" and is like a smoothed
  version of the Heaviside (Unit Step Function) function. For small values
  (<-5) the sigmoid returns a value close to zero and for larger values (>5)
  the result of the function gets close to 1.
  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  Sigmoid activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The sigmoid activation: `(1.0 / (1.0 + exp(-x)))`.
  "
  [ x ]
  (py/call-attr activations "sigmoid"  x ))
(defn softmax 
  "The softmax activation function transforms the outputs so that all values are in

  range (0, 1) and sum to 1. It is often used as the activation for the last
  layer of a classification network because the result could be interpreted as
  a probability distribution. The softmax of x is calculated by
  exp(x)/tf.reduce_sum(exp(x)).

  Arguments:
      x : Input tensor.
      axis: Integer, axis along which the softmax normalization is applied.

  Returns:
      Tensor, output of softmax transformation (all values are non-negative
        and sum to 1).

  Raises:
      ValueError: In case `dim(x) == 1`.
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw activations "softmax" [x] {:axis axis }))

(defn softplus 
  "Softplus activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `log(exp(x) + 1)`.
  "
  [ x ]
  (py/call-attr activations "softplus"  x ))

(defn softsign 
  "Softsign activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `x / (abs(x) + 1)`.
  "
  [ x ]
  (py/call-attr activations "softsign"  x ))

(defn tanh 
  "Hyperbolic Tangent (tanh) activation function.

  For example:

  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32)
  b = tf.keras.activations.tanh(a) #[-0.9950547,-0.7615942,
  0.,0.7615942,0.9950547]
  ```
  Arguments:
      x: Input tensor.

  Returns:
      A tensor of same shape and dtype of input `x`.
      The tanh activation: `tanh(x) = sinh(x)/cosh(x) = ((exp(x) -
      exp(-x))/(exp(x) + exp(-x)))`.
  "
  [ x ]
  (py/call-attr activations "tanh"  x ))
