# nabo

[![Crates.io][crates-badge]][crates-url]
[![Docs.rs][docs-badge]][docs-url]
[![Build Status][ci-badge]][ci-url]

[crates-badge]: https://img.shields.io/crates/v/nabo
[crates-url]: https://crates.io/crates/nabo
[docs-badge]: https://img.shields.io/docsrs/nabo
[docs-url]: https://docs.rs/nabo
[ci-badge]: https://github.com/enlightware/nabo-rs/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/enlightware/nabo-rs/actions

## Overview

nabo is a fast K Nearest Neighbour (KNN) library for low-dimensional spaces.
It is a re-implementation in pure Rust of the [C++ library of the same name](https://github.com/ethz-asl/libnabo) by its [original author](http://stephane.magnenat.net).
This work has been sponsored by [Enlightware GmbH](https://enlightware.ch).

## Usage

To use nabo in your project, you need either:
- Use `src/simple_point`.
- Implement the `nabo::Point` trait for your point type.

If you want to avoid a dependency to `rand`, disable the `rand` feature.

## Benchmark

You can benchmark nabo using the following command:

    cargo bench

## Citing nabo

If you use nabo in the academic context, please cite this paper that evaluates its performances in the context of robotics mapping research:

	@article{elsebergcomparison,
		title={Comparison of nearest-neighbor-search strategies and implementations for efficient shape registration},
		author={Elseberg, J. and Magnenat, S. and Siegwart, R. and N{\"u}chter, A.},
		journal={Journal of Software Engineering for Robotics (JOSER)},
		pages={2--12},
		volume={3},
		number={1},
		year={2012},
		issn={2035-3928}
	}

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
