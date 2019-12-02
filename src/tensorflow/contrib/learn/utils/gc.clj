(ns tensorflow.contrib.learn.python.learn.utils.gc
  "System for specifying garbage collection (GC) of path based data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

This framework allows for GC of data specified by path names, for example files
on disk.  gc.Path objects each represent a single item stored at a path and may
be a base directory,
  /tmp/exports/0/...
  /tmp/exports/1/...
  ...
or a fully qualified file,
  /tmp/train-1.ckpt
  /tmp/train-2.ckpt
  ...

A gc filter function takes and returns a list of gc.Path items.  Filter
functions are responsible for selecting Path items for preservation or deletion.
Note that functions should always return a sorted list.

For example,
  base_dir = \"/tmp\"
  # Create the directories.
  for e in xrange(10):
    os.mkdir(\"%s/%d\" % (base_dir, e), 0o755)

  # Create a simple parser that pulls the export_version from the directory.
  path_regex = \"^\" + re.escape(base_dir) + \"/(\\d+)$\"
  def parser(path):
    match = re.match(path_regex, path.path)
    if not match:
      return None
    return path._replace(export_version=int(match.group(1)))

  path_list = gc.get_paths(\"/tmp\", parser)  # contains all ten Paths

  every_fifth = gc.mod_export_version(5)
  print(every_fifth(path_list))  # shows [\"/tmp/0\", \"/tmp/5\"]

  largest_three = gc.largest_export_versions(3)
  print(largest_three(all_paths))  # shows [\"/tmp/7\", \"/tmp/8\", \"/tmp/9\"]

  both = gc.union(every_fifth, largest_three)
  print(both(all_paths))  # shows [\"/tmp/0\", \"/tmp/5\",
                          #        \"/tmp/7\", \"/tmp/8\", \"/tmp/9\"]
  # Delete everything not in 'both'.
  to_delete = gc.negation(both)
  for p in to_delete(all_paths):
    gfile.rmtree(p.path)  # deletes:  \"/tmp/1\", \"/tmp/2\",
                                     # \"/tmp/3\", \"/tmp/4\", \"/tmp/6\",
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gc (import-module "tensorflow.contrib.learn.python.learn.utils.gc"))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw gc "deprecated" [date instructions] {:warn_once warn_once }))

(defn get-paths 
  "Gets a list of Paths in a given directory. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file name management.

Args:
  base_dir: directory.
  parser: a function which gets the raw Path and can augment it with
    information such as the export_version, or ignore the path by returning
    None.  An example parser may extract the export version from a path
    such as \"/tmp/exports/100\" an another may extract from a full file
    name such as \"/tmp/checkpoint-99.out\".

Returns:
  A list of Paths contained in the base directory with the parsing function
  applied.
  By default the following fields are populated,
    - Path.path
  The parsing function is responsible for populating,
    - Path.export_version"
  [ base_dir parser ]
  (py/call-attr gc "get_paths"  base_dir parser ))

(defn largest-export-versions 
  "Creates a filter that keeps the largest n export versions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file management or use Saver.

Args:
  n: number of versions to keep.

Returns:
  A filter function that keeps the n largest paths."
  [ n ]
  (py/call-attr gc "largest_export_versions"  n ))

(defn mod-export-version 
  "Creates a filter that keeps every export that is a multiple of n. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file management or use Saver.

Args:
  n: step size.

Returns:
  A filter function that keeps paths where export_version % n == 0."
  [ n ]
  (py/call-attr gc "mod_export_version"  n ))

(defn negation 
  "Negate a filter. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file management or use Saver.

Args:
  f: filter function to invert

Returns:
  A filter function that returns the negation of f."
  [ f ]
  (py/call-attr gc "negation"  f ))

(defn one-of-every-n-export-versions 
  "Creates a filter that keeps one of every n export versions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file management or use Saver.

Args:
  n: interval size.

Returns:
  A filter function that keeps exactly one path from each interval
  [0, n], (n, 2n], (2n, 3n], etc...  If more than one path exists in an
  interval the largest is kept."
  [ n ]
  (py/call-attr gc "one_of_every_n_export_versions"  n ))

(defn union 
  "Creates a filter that keeps the union of two filters. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please implement your own file management or use Saver.

Args:
  lf: first filter
  rf: second filter

Returns:
  A filter function that keeps the n largest paths."
  [ lf rf ]
  (py/call-attr gc "union"  lf rf ))
