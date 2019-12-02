(ns tensorflow.-api.v1.compat.v1.nest
  "Public API for tf.nest namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nest (import-module "tensorflow._api.v1.compat.v1.nest"))
(defn assert-same-structure 
  "Asserts that two structures are nested in the same way.

  Note that namedtuples with identical name and fields are always considered
  to have the same shallow structure (even with `check_types=True`).
  For instance, this code will print `True`:

  ```python
  def nt(a, b):
    return collections.namedtuple('foo', 'a b')(a, b)
  print(assert_same_structure(nt(0, 1), nt(2, 3)))
  ```

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences are checked as well,
        including the keys of dictionaries. If set to `False`, for example a
        list and a tuple of objects will look the same if they have the same
        size. Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure. Two types will also be
        considered the same if they are both list subtypes (which allows \"list\"
        and \"_ListWrapper\" from trackable dependency tracking to compare
        equal).
    expand_composites: If true, then composite tensors such as `tf.SparseTensor`
        and `tf.RaggedTensor` are expanded into their component tensors.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  "
  [nest1 nest2  & {:keys [check_types expand_composites]} ]
    (py/call-attr-kw nest "assert_same_structure" [nest1 nest2] {:check_types check_types :expand_composites expand_composites }))
(defn flatten 
  "Returns a flat list from a given nested structure.

  If nest is not a sequence, tuple, or dict, then returns a single-element list:
  [nest].

  In the case of dict instances, the sequence consists of the values, sorted by
  key to ensure deterministic behavior. This is true also for OrderedDict
  instances: their sequence order is ignored, the sorting order of keys is used
  instead. The same convention is followed in pack_sequence_as. This correctly
  repacks dicts and OrderedDicts after they have been flattened, and also allows
  flattening an OrderedDict and then repacking it back using a corresponding
  plain dict, or vice-versa. Dictionaries with non-sortable keys cannot be
  flattened.

  Users must not modify any collections used in nest while this function is
  running.

  Args:
    structure: an arbitrarily nested structure or a scalar object. Note, numpy
      arrays are considered scalars.
    expand_composites: If true, then composite tensors such as tf.SparseTensor
       and tf.RaggedTensor are expanded into their component tensors.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  "
  [structure  & {:keys [expand_composites]} ]
    (py/call-attr-kw nest "flatten" [structure] {:expand_composites expand_composites }))

(defn is-nested 
  "Returns true if its input is a collections.abc.Sequence (except strings).

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a not a string and is a collections.abc.Sequence
    or a dict.
  "
  [ seq ]
  (py/call-attr nest "is_nested"  seq ))

(defn map-structure 
  "Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain results with the same structure layout.

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structure: scalar, or tuple or list of constructed scalars and/or other
      tuples/lists, or scalars.  Note: numpy arrays are considered as scalars.
    **kwargs: Valid keyword args are:

      * `check_types`: If set to `True` (default) the types of
        iterables within the structures have to be same (e.g.
        `map_structure(func, [1], (1,))` raises a `TypeError`
        exception). To allow this set this argument to `False`.
        Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure.
      * `expand_composites`: If set to `True`, then composite tensors such
        as `tf.SparseTensor` and `tf.RaggedTensor` are expanded into their
        component tensors.  If `False` (the default), then composite tensors
        are not expanded.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`. If there are different sequence types and
    `check_types` is `False` the sequence types of the first structure will be
    used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
    ValueError: If wrong keyword arguments are provided.
  "
  [ func ]
  (py/call-attr nest "map_structure"  func ))
(defn pack-sequence-as 
  "Returns a given flattened sequence packed into a given structure.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  If `structure` is or contains a dict instance, the keys will be sorted to
  pack the flat sequence in deterministic order. This is true also for
  `OrderedDict` instances: their sequence order is ignored, the sorting order of
  keys is used instead. The same convention is followed in `flatten`.
  This correctly repacks dicts and `OrderedDict`s after they have been
  flattened, and also allows flattening an `OrderedDict` and then repacking it
  back using a corresponding plain dict, or vice-versa.
  Dictionaries with non-sortable keys cannot be flattened.

  Args:
    structure: Nested structure, whose structure is given by nested lists,
        tuples, and dicts. Note: numpy arrays and strings are considered
        scalars.
    flat_sequence: flat sequence to pack.
    expand_composites: If true, then composite tensors such as `tf.SparseTensor`
        and `tf.RaggedTensor` are expanded into their component tensors.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      element counts.
    TypeError: `structure` is or contains a dict with non-sortable keys.
  "
  [structure flat_sequence  & {:keys [expand_composites]} ]
    (py/call-attr-kw nest "pack_sequence_as" [structure flat_sequence] {:expand_composites expand_composites }))
