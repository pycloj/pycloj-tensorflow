(ns tensorflow.-api.v1.compat.v1.gfile
  "Import router for file_io.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gfile (import-module "tensorflow._api.v1.compat.v1.gfile"))
(defn Copy 
  "Copies data from `oldpath` to `newpath`.

  Args:
    oldpath: string, name of the file who's contents need to be copied
    newpath: string, name of the file to which to copy to
    overwrite: boolean, if false it's an error for `newpath` to be occupied by
      an existing file.

  Raises:
    errors.OpError: If the operation fails.
  "
  [oldpath newpath  & {:keys [overwrite]} ]
    (py/call-attr-kw gfile "Copy" [oldpath newpath] {:overwrite overwrite }))

(defn DeleteRecursively 
  "Deletes everything under dirname recursively.

  Args:
    dirname: string, a path to a directory

  Raises:
    errors.OpError: If the operation fails.
  "
  [ dirname ]
  (py/call-attr gfile "DeleteRecursively"  dirname ))

(defn Exists 
  "Determines whether a path exists or not.

  Args:
    filename: string, a path

  Returns:
    True if the path exists, whether it's a file or a directory.
    False if the path does not exist and there are no filesystem errors.

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  "
  [ filename ]
  (py/call-attr gfile "Exists"  filename ))

(defn Glob 
  "Returns a list of files that match the given pattern(s).

  Args:
    filename: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
  "
  [ filename ]
  (py/call-attr gfile "Glob"  filename ))

(defn IsDirectory 
  "Returns whether the path is a directory or not.

  Args:
    dirname: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  "
  [ dirname ]
  (py/call-attr gfile "IsDirectory"  dirname ))

(defn ListDirectory 
  "Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries \".\"
  and \"..\".

  Args:
    dirname: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN] as strings

  Raises:
    errors.NotFoundError if directory doesn't exist
  "
  [ dirname ]
  (py/call-attr gfile "ListDirectory"  dirname ))

(defn MakeDirs 
  "Creates a directory and all parent/intermediate directories.

  It succeeds if dirname already exists and is writable.

  Args:
    dirname: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  "
  [ dirname ]
  (py/call-attr gfile "MakeDirs"  dirname ))

(defn MkDir 
  "Creates a directory with the name 'dirname'.

  Args:
    dirname: string, name of the directory to be created
  Notes: The parent directories need to exist. Use recursive_create_dir instead
    if there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  "
  [ dirname ]
  (py/call-attr gfile "MkDir"  dirname ))

(defn Remove 
  "Deletes the file located at 'filename'.

  Args:
    filename: string, a filename

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    NotFoundError if the file does not exist.
  "
  [ filename ]
  (py/call-attr gfile "Remove"  filename ))
(defn Rename 
  "Rename or move a file / directory.

  Args:
    oldname: string, pathname for a file
    newname: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `newname` to be occupied by
      an existing file.

  Raises:
    errors.OpError: If the operation fails.
  "
  [oldname newname  & {:keys [overwrite]} ]
    (py/call-attr-kw gfile "Rename" [oldname newname] {:overwrite overwrite }))

(defn Stat 
  "Returns file statistics for a given path.

  Args:
    filename: string, path to a file

  Returns:
    FileStatistics struct that contains information about the path

  Raises:
    errors.OpError: If the operation fails.
  "
  [ filename ]
  (py/call-attr gfile "Stat"  filename ))
(defn Walk 
  "Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    in_order: bool, Traverse in order if True, post order if False.  Errors that
      happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
    as strings
  "
  [top  & {:keys [in_order]} ]
    (py/call-attr-kw gfile "Walk" [top] {:in_order in_order }))
