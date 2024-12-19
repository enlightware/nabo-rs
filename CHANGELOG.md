### Unreleased

### 0.4.0 - 0224-12-20

* Made `nabo` `no_std` compatible.
* Renamed `dummy_point` as `simple_point`.
* Added functions to iterate over the points of the tree.

### 0.3.0 - 2023-06-28

* Derive `Clone` trait for `KDTree`.
* Switched to Rust 2021 edition.
* Updated `ordered-float` to version 3.7 and `criterion` to version 0.5.

### 0.2.1 - 2021-09-13

* Fixed linear candidate container to return less than k entries rather than entries with infinity if not enough neighbours can be found.

### 0.2.0 - 2021-09-06

* Improved documentation.
* Made naming of functions in `dummy_point` and unit tests consistent.

### 0.1.0 - 2021-09-03

* Initial re-implementation of the C++ library and adaptation to Rust.
