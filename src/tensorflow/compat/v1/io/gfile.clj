(ns tensorflow.-api.v1.compat.v1.io.gfile
  "Public API for tf.io.gfile namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gfile (import-module "tensorflow._api.v1.compat.v1.io.gfile"))
(defn copy 
  "Copies data from `src` to `dst`.

  Args:
    src: string, name of the file whose contents need to be copied
    dst: string, name of the file to which to copy to
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.

  Raises:
    errors.OpError: If the operation fails.
  "
  [src dst  & {:keys [overwrite]} ]
    (py/call-attr-kw gfile "copy" [src dst] {:overwrite overwrite }))

(defn exists 
  "Determines whether a path exists or not.

  Args:
    path: string, a path

  Returns:
    True if the path exists, whether it's a file or a directory.
    False if the path does not exist and there are no filesystem errors.

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  "
  [ path ]
  (py/call-attr gfile "exists"  path ))

(defn glob 
  "Returns a list of files that match the given pattern(s).

  Args:
    pattern: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
  "
  [ pattern ]
  (py/call-attr gfile "glob"  pattern ))

(defn isdir 
  "Returns whether the path is a directory or not.

  Args:
    path: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  "
  [ path ]
  (py/call-attr gfile "isdir"  path ))

(defn listdir 
  "Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries \".\"
  and \"..\".

  Args:
    path: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN] as strings

  Raises:
    errors.NotFoundError if directory doesn't exist
  "
  [ path ]
  (py/call-attr gfile "listdir"  path ))

(defn makedirs 
  "Creates a directory and all parent/intermediate directories.

  It succeeds if path already exists and is writable.

  Args:
    path: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  "
  [ path ]
  (py/call-attr gfile "makedirs"  path ))

(defn mkdir 
  "Creates a directory with the name given by 'path'.

  Args:
    path: string, name of the directory to be created
  Notes: The parent directories need to exist. Use recursive_create_dir instead
    if there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  "
  [ path ]
  (py/call-attr gfile "mkdir"  path ))

(defn remove 
  "Deletes the path located at 'path'.

  Args:
    path: string, a path

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    NotFoundError if the path does not exist.
  "
  [ path ]
  (py/call-attr gfile "remove"  path ))
(defn rename 
  "Rename or move a file / directory.

  Args:
    src: string, pathname for a file
    dst: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.

  Raises:
    errors.OpError: If the operation fails.
  "
  [src dst  & {:keys [overwrite]} ]
    (py/call-attr-kw gfile "rename" [src dst] {:overwrite overwrite }))

(defn rmtree 
  "Deletes everything under path recursively.

  Args:
    path: string, a path

  Raises:
    errors.OpError: If the operation fails.
  "
  [ path ]
  (py/call-attr gfile "rmtree"  path ))

(defn stat 
  "Returns file statistics for a given path.

  Args:
    path: string, path to a file

  Returns:
    FileStatistics struct that contains information about the path

  Raises:
    errors.OpError: If the operation fails.
  "
  [ path ]
  (py/call-attr gfile "stat"  path ))

(defn walk 
  "Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    topdown: bool, Traverse pre order if True, post order if False.
    onerror: optional handler for errors. Should be a function, it will be
      called with the error as argument. Rethrowing the error aborts the walk.
      Errors that happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
    as strings
  "
  [top & {:keys [topdown onerror]
                       :or {onerror None}} ]
    (py/call-attr-kw gfile "walk" [top] {:topdown topdown :onerror onerror }))
