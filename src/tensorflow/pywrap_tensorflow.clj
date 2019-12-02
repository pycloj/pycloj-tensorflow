(ns tensorflow-core.python.pywrap-tensorflow
  "A wrapper for TensorFlow SWIG-generated bindings."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pywrap-tensorflow (import-module "tensorflow_core.python.pywrap_tensorflow"))
(defn Flatten 
  "
    Returns a flat list from a given nested structure.

    If `nest` is not a sequence, tuple, or dict, then returns a single-element
    list: `[nest]`.

    In the case of dict instances, the sequence consists of the values, sorted by
    key to ensure deterministic behavior. This is true also for `OrderedDict`
    instances: their sequence order is ignored, the sorting order of keys is
    used instead. The same convention is followed in `pack_sequence_as`. This
    correctly repacks dicts and `OrderedDict`s after they have been flattened,
    and also allows flattening an `OrderedDict` and then repacking it back using
    a corresponding plain dict, or vice-versa.
    Dictionaries with non-sortable keys cannot be flattened.

    Users must not modify any collections used in `nest` while this function is
    running.

    Args:
      nest: an arbitrarily nested structure or a scalar object. Note, numpy
          arrays are considered scalars.
      expand_composites: If true, then composite tensors such as `tf.SparseTensor`
          and `tf.RaggedTensor` are expanded into their component tensors.

    Returns:
      A Python list, the flattened version of the input.

    Raises:
      TypeError: The nest is or contains a dict with non-sortable keys.

    "
  [nested  & {:keys [expand_composites]} ]
    (py/call-attr-kw pywrap-tensorflow "Flatten" [nested] {:expand_composites expand_composites }))

(defn FlattenForData 
  "
    Returns a flat sequence from a given nested structure.

    If `nest` is not a sequence, this returns a single-element list: `[nest]`.

    Args:
      nest: an arbitrarily nested structure or a scalar object.
        Note, numpy arrays are considered scalars.

    Returns:
      A Python list, the flattened version of the input.

    "
  [ nested ]
  (py/call-attr pywrap-tensorflow "FlattenForData"  nested ))

(defn IsAttrs 
  "
    Returns True iff `instance` is an instance of an `attr.s` decorated class.

    Args:
      instance: An instance of a Python object.

    Returns:
      True if `instance` is an instance of an `attr.s` decorated class.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsAttrs"  o ))

(defn IsCompositeTensor 
  "
    Returns true if its input is a `CompositeTensor`.

    Args:
      seq: an input sequence.

    Returns:
      True if the sequence is a CompositeTensor.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsCompositeTensor"  o ))

(defn IsMapping 
  "
    Returns True iff `instance` is a `collections.Mapping`.

    Args:
      instance: An instance of a Python object.

    Returns:
      True if `instance` is a `collections.Mapping`.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsMapping"  o ))

(defn IsMappingView 
  "
    Returns True iff `instance` is a `collections.MappingView`.

    Args:
      instance: An instance of a Python object.

    Returns:
      True if `instance` is a `collections.MappingView`.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsMappingView"  o ))

(defn IsSequence 
  "
    Returns true if its input is a collections.Sequence (except strings).

    Args:
      seq: an input sequence.

    Returns:
      True if the sequence is a not a string and is a collections.Sequence or a
      dict.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsSequence"  o ))

(defn IsSequenceForData 
  "
    Returns a true if `seq` is a Sequence or dict (except strings/lists).

    NOTE(mrry): This differs from `tensorflow.python.util.nest.is_sequence()`,
    which *does* treat a Python list as a sequence. For ergonomic
    reasons, `tf.data` users would prefer to treat lists as
    implicit `tf.Tensor` objects, and dicts as (nested) sequences.

    Args:
      seq: an input sequence.

    Returns:
      True if the sequence is a not a string or list and is a
      collections.Sequence.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsSequenceForData"  o ))

(defn IsSequenceOrComposite 
  "
    Returns true if its input is a sequence or a `CompositeTensor`.

    Args:
      seq: an input sequence.

    Returns:
      True if the sequence is a not a string and is a collections.Sequence or a
      dict or a CompositeTensor or a TypeSpec (except string and TensorSpec).

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsSequenceOrComposite"  o ))

(defn IsTypeSpec 
  "
    Returns true if its input is a `TypeSpec`, but is not a `TensorSpec`.

    Args:
      seq: an input sequence.

    Returns:
      True if the sequence is a `TypeSpec`, but is not a `TensorSpec`.

    "
  [ o ]
  (py/call-attr pywrap-tensorflow "IsTypeSpec"  o ))

(defn NewCheckpointReader 
  ""
  [ filepattern ]
  (py/call-attr pywrap-tensorflow "NewCheckpointReader"  filepattern ))

(defn SameNamedtuples 
  "Returns True if the two namedtuples have the same name and fields."
  [ o1 o2 ]
  (py/call-attr pywrap-tensorflow "SameNamedtuples"  o1 o2 ))

(defn TF-NewSessionOptions 
  ""
  [ target config ]
  (py/call-attr pywrap-tensorflow "TF_NewSessionOptions"  target config ))

(defn TF-Reset 
  ""
  [ target containers config ]
  (py/call-attr pywrap-tensorflow "TF_Reset"  target containers config ))

(defn do-quantize-training-on-graphdef 
  "A general quantization scheme is being developed in `tf.contrib.quantize`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
GraphDef quantized training rewriter is deprecated in the long term

Consider using that instead, though since it is in the tf.contrib namespace,
it is not subject to backward compatibility guarantees."
  [ input_graph num_bits ]
  (py/call-attr pywrap-tensorflow "do_quantize_training_on_graphdef"  input_graph num_bits ))

(defn list-devices 
  ""
  [ session_config ]
  (py/call-attr pywrap-tensorflow "list_devices"  session_config ))
