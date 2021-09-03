#![warn(missing_docs)]

//! A fast K Nearest Neighbour library for low-dimensional spaces.
//!
//! This crate is a  re-implementation in pure Rust of the [C++ library of the same name](https://github.com/ethz-asl/libnabo).
//! This work has been sponsored by [Enlightware GmbH](https://enlightware.ch).
//!
//! # Example
//! ```
//! use nabo::dummy_point::*;
//! use nabo::KDTree;
//! let cloud = cloud_random(10000);
//! let tree = KDTree::new(&cloud);
//! let query = rand_point();
//! let neighbour = tree.knn(3, &query);
//! ```
//!
//! If you want to have more control, on the search, you can use the advance API:
//! ```
//! use nabo::dummy_point::*;
//! use nabo::KDTree;
//! use nabo::CandidateContainer;
//! use nabo::Parameters;
//! let cloud = cloud_random(10000);
//! let tree = KDTree::new(&cloud);
//! let query = rand_point();
//! let mut touch_count = 0;
//! let neighbour = tree.knn_advanced(
//!     3,
//!     &query,
//!     CandidateContainer::BinaryHeap,
//!     &Parameters {
//!         epsilon: 0.0,
//!         max_radius: 10.0,
//!         allow_self_match: true,
//!         sort_results: false,
//!     },
//!     Some(&mut touch_count) // statistics
//! );
//! ```

// We forbid the clippy lint here because it suggests to use #[rustfmt::skip],
// which is experimental. See: https://github.com/rust-lang/rust/issues/88591
#![allow(clippy::deprecated_cfg_attr)]

#[cfg(any(test, feature = "dummy_point"))]
pub mod dummy_point;
mod heap;
mod infinite;
mod internal_neighbour;
mod internal_parameters;
mod node;

use internal_parameters::InternalParameters;
use node::Node;
use num_traits::{clamp_max, clamp_min, Bounded, Zero};
use ordered_float::Float;
pub use ordered_float::NotNan;
use std::{collections::BinaryHeap, ops::AddAssign};

use heap::CandidateHeap;
use internal_neighbour::InternalNeighbour;

/// The scalar type for points in the space to be searched
pub trait Scalar: Float + AddAssign + std::fmt::Debug {}
impl<T: Float + AddAssign + std::fmt::Debug> Scalar for T {}

/// A point in the space to be searched
pub trait Point<T: Scalar>: Default {
    /// Sets the value for a given index, which must be within `0..DIM`.
    fn set(&mut self, i: u32, value: NotNan<T>);
    /// Gets the value for a given index, which must be within `0..DIM`.
    fn get(&self, i: u32) -> NotNan<T>;
    /// The number of dimension of the space this point lies in.
    const DIM: u32;
    /// Derived from `DIM`, do not reimplement, use the default!
    const DIM_BIT_COUNT: u32 = 32 - Self::DIM.leading_zeros();
    /// Derived from `DIM`, do not reimplement, use the default!
    const DIM_MASK: u32 = (1 << Self::DIM_BIT_COUNT) - 1;
    /// Derived from `DIM`, do not reimplement, use the default!
    const MAX_NODE_COUNT: u32 = ((1u64 << (32 - Self::DIM_BIT_COUNT)) - 1) as u32;
}

/// Helper function to compute the square distance between two points given as slice
#[inline]
fn slice_dist2<T: Scalar, P: Point<T>>(lhs: &[NotNan<T>], rhs: &[NotNan<T>]) -> NotNan<T> {
    let mut dist2 = NotNan::<T>::zero();
    for index in 0..P::DIM {
        let index = index as usize;
        let diff = lhs[index] - rhs[index];
        dist2 += diff * diff;
    }
    dist2
}

/// The index of a point in the original point cloud
pub type Index = u32;

/// A neighbour resulting from the search
pub struct Neighbour<T: Scalar, P: Point<T>> {
    /// the point itself
    pub point: P,
    /// the squared-distance to the point
    pub dist2: NotNan<T>,
    /// the index of the point in the original point cloud
    pub index: Index,
}

/// The type of container to keep candidates
pub enum CandidateContainer {
    /// use a linear vector to keep candidates, good for small k
    Linear,
    /// use a binary heap to keep candidates, good for large k
    BinaryHeap,
}

/// Advanced search parameters
pub struct Parameters<T: Scalar> {
    /// maximal ratio of error for approximate search, 0 for exact search; has no effect if the number of neighbours found is smaller than the number requested
    pub epsilon: T,
    /// maximum radius in which to search, can be used to prune search, is not affected by `epsilon`
    pub max_radius: T,
    /// allows the return of the same point as the query, if this point is in the point cloud
    pub allow_self_match: bool,
    /// sort points by distances, when `k` > 1
    pub sort_results: bool,
}

/// A dense vector of search nodes, provides better memory performances than many small objects
type Nodes<T, P> = Vec<Node<T, P>>;

/// A KD-Tree to perform NN-search queries
///
/// This implementation is inspired of the variant `KDTreeUnbalancedPtInLeavesImplicitBoundsStackOpt` in libnabo C++.
/// Contrary to the latter, it does not keep a reference to the point cloud but copies the point.
/// It retains their index though.
#[derive(Debug)]
pub struct KDTree<T: Scalar, P: Point<T>> {
    /// size of a bucket
    bucket_size: u32,
    /// search nodes
    nodes: Nodes<T, P>,
    /// point data, size cloud.len() * P::DIM
    points: Vec<NotNan<T>>,
    /// indices in cloud , size cloud.len()
    indices: Vec<Index>,
}

impl<T: Scalar, P: Point<T>> KDTree<T, P> {
    /// Creates a new KD-Tree from a point cloud.
    pub fn new(cloud: &[P]) -> Self {
        KDTree::new_with_bucket_size(cloud, 8)
    }
    /// Creates a new KD-Tree from a point cloud.
    ///
    /// The `bucket_size` can be chosen freely, but must be at least 2.
    pub fn new_with_bucket_size(cloud: &[P], bucket_size: u32) -> Self {
        // validate input
        if bucket_size < 2 {
            panic!(
                "Bucket size must be at least 2, but {} was passed",
                bucket_size
            );
        }
        if cloud.len() > u32::MAX as usize {
            panic!(
                "Point cloud is larger than maximum possible size {}",
                u32::MAX
            );
        }
        let estimated_node_count = (cloud.len() / (bucket_size as usize / 2)) as u32;
        if estimated_node_count > P::MAX_NODE_COUNT {
            panic!("Point cloud has a risk to have more nodes {} than the kd-tree allows {}. The kd-tree has {} bits for dimensions and {} bits for node indices", estimated_node_count, P::MAX_NODE_COUNT, P::DIM_BIT_COUNT, 32 - P::DIM_BIT_COUNT);
        }

        // build point vector and compute bounds
        let mut build_points: Vec<_> = (0..cloud.len()).collect();

        // create and populate tree
        let mut tree = KDTree {
            bucket_size,
            nodes: Vec::with_capacity(estimated_node_count as usize),
            points: Vec::with_capacity(cloud.len() * P::DIM as usize),
            indices: Vec::with_capacity(cloud.len()),
        };
        tree.build_nodes(cloud, &mut build_points);
        tree
    }

    /// Finds the `k` nearest neighbour of `query`, using reasonable default parameters.
    ///
    /// The default parameters are:
    /// Exact search, no max. radius, allowing self matching, sorting results, and not collecting statistics.
    /// If `k` <= 16, a linear vector is used to keep track of candidates, otherwise a binary heap is used.
    pub fn knn(&self, k: u32, query: &P) -> Vec<Neighbour<T, P>> {
        let candidate_container = if k <= 16 {
            CandidateContainer::Linear
        } else {
            CandidateContainer::BinaryHeap
        };
        #[cfg_attr(rustfmt, rustfmt_skip)]
        self.knn_advanced(
            k, query,
            candidate_container,
            &Parameters {
                epsilon: T::from(0.0).unwrap(),
                max_radius: T::infinity(),
                allow_self_match: true,
                sort_results: true,
            },
            None,
        )
    }

    /// Finds the `k` nearest neighbour of `query`, with user-provided parameters.
    ///
    /// The parameters are:
    /// * `candidate_container` which container to use to collect candidates,
    /// * `parameters` the advanced search parameters,
    /// * `touch_statistics`, if `Some(&mut u32)`, return the number of point touched in the provided `u32` reference.
    pub fn knn_advanced(
        &self,
        k: u32,
        query: &P,
        candidate_container: CandidateContainer,
        parameters: &Parameters<T>,
        touch_statistics: Option<&mut u32>,
    ) -> Vec<Neighbour<T, P>> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        (match candidate_container {
            CandidateContainer::Linear => Self::knn_generic_heap::<Vec<InternalNeighbour<T>>>,
            CandidateContainer::BinaryHeap => Self::knn_generic_heap::<BinaryHeap<InternalNeighbour<T>>>
        })(
            self,
            k, query,
            parameters, touch_statistics
        )
    }

    fn knn_generic_heap<H: CandidateHeap<T>>(
        &self,
        k: u32,
        query: &P,
        parameters: &Parameters<T>,
        touch_statistics: Option<&mut u32>,
    ) -> Vec<Neighbour<T, P>> {
        let query_as_vec: Vec<_> = (0..P::DIM).map(|i| query.get(i)).collect();
        let Parameters {
            epsilon,
            max_radius,
            allow_self_match,
            sort_results,
        } = *parameters;
        let max_error = epsilon + T::from(1).unwrap();
        let max_error2 = NotNan::new(max_error * max_error).unwrap();
        let max_radius2 = NotNan::new(max_radius * max_radius).unwrap();
        #[cfg_attr(rustfmt, rustfmt_skip)]
        self.knn_internal::<H>(
            k, &query_as_vec,
            &InternalParameters { max_error2, max_radius2, allow_self_match },
            sort_results, touch_statistics,
        )
            .into_iter()
            .map(|n| self.externalise_neighbour(n))
            .collect()
    }

    fn knn_internal<H: CandidateHeap<T>>(
        &self,
        k: u32,
        query: &[NotNan<T>],
        internal_parameters: &InternalParameters<T>,
        sort_results: bool,
        touch_statistics: Option<&mut u32>,
    ) -> Vec<InternalNeighbour<T>> {
        // TODO Const generics: once available, remove `vec!` below.
        let mut off = vec![NotNan::<T>::zero(); P::DIM as usize];
        let mut heap = H::new_with_k(k);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        let leaf_touched_count = self.recurse_knn(
            k, query,
            0, NotNan::<T>::zero(),
            &mut heap, &mut off,
            internal_parameters,
        );
        if let Some(touch_statistics) = touch_statistics {
            *touch_statistics = leaf_touched_count;
        }
        if sort_results {
            heap.into_sorted_vec()
        } else {
            heap.into_vec()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn recurse_knn<H: CandidateHeap<T>>(
        &self,
        k: u32,
        query: &[NotNan<T>],
        node: usize,
        rd: NotNan<T>,
        heap: &mut H,
        off: &mut [NotNan<T>],
        internal_parameters: &InternalParameters<T>,
    ) -> u32 {
        self.nodes[node].dispatch_on_type(
            heap,
            |heap, split_dim, split_val, right_child| {
                // split node, see whether we have to recurse
                let mut rd = rd;
                let split_dim = split_dim as usize;
                let old_off = off[split_dim];
                let new_off = query[split_dim] - split_val;
                let left_child = node + 1;
                let right_child = right_child as usize;
                let InternalParameters {
                    max_radius2,
                    max_error2,
                    ..
                } = *internal_parameters;
                if new_off > NotNan::<T>::zero() {
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    let mut leaf_visited_count = self.recurse_knn(
                        k, query,
                        right_child, rd,
                        heap, off,
                        internal_parameters,
                    );
                    rd += new_off * new_off - old_off * old_off;
                    if rd <= max_radius2 && rd * max_error2 < heap.furthest_dist2() {
                        off[split_dim] = new_off;
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        let new_visits= self.recurse_knn(
                            k, query,
                            left_child, rd,
                            heap, off,
                            internal_parameters,
                        );
                        leaf_visited_count += new_visits;
                        off[split_dim] = old_off;
                    }
                    leaf_visited_count
                } else {
                    #[cfg_attr(rustfmt, rustfmt_skip)]
                    let mut leaf_visited_count = self.recurse_knn(
                        k, query,
                        left_child, rd,
                        heap, off,
                        internal_parameters,
                    );
                    rd += new_off * new_off - old_off * old_off;
                    if rd <= max_radius2 && rd * max_error2 < heap.furthest_dist2() {
                        off[split_dim] = new_off;
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        let new_visits = self.recurse_knn(
                            k, query,
                            right_child, rd,
                            heap, off,
                            internal_parameters,
                        );
                        leaf_visited_count += new_visits;
                        off[split_dim] = old_off;
                    }
                    leaf_visited_count
                }
            },
            |heap, bucket_start_index, bucket_size| {
                // leaf node, go through the buckets and check elements
                let bucket_end_index = bucket_start_index + bucket_size;
                for bucket_index in bucket_start_index..bucket_end_index {
                    let point_index = (bucket_index * P::DIM) as usize;
                    let point = &self.points[point_index..point_index + (P::DIM as usize)];
                    let dist2 = slice_dist2::<T, P>(query, point);
                    let epsilon = NotNan::new(T::epsilon()).unwrap();
                    let InternalParameters {
                        max_radius2,
                        allow_self_match,
                        ..
                    } = *internal_parameters;
                    if dist2 < max_radius2 && (allow_self_match || (dist2 > epsilon)) {
                        heap.add(dist2, bucket_index);
                    }
                }
                bucket_size
            },
        )
    }

    fn build_nodes(&mut self, cloud: &[P], build_points: &mut [usize]) -> usize {
        let count = build_points.len() as u32;
        let pos = self.nodes.len();

        // if remaining points fit in a single bucket, add a node and this bucket
        if count <= self.bucket_size {
            let bucket_start_index = self.indices.len() as u32;
            self.points.reserve(build_points.len() * P::DIM as usize);
            self.indices.reserve(build_points.len());
            for point_index in build_points {
                let point_index = *point_index;
                self.indices.push(point_index as u32);
                for i in 0..P::DIM {
                    self.points.push(cloud[point_index].get(i));
                }
            }
            self.nodes
                .push(Node::new_leaf_node(bucket_start_index, count));
            return pos;
        }

        // compute bounds
        let (min_bounds, max_bounds) = Self::get_build_points_bounds(cloud, build_points);

        // find the largest dimension of the box
        let split_dim = Self::max_delta_index(&min_bounds, &max_bounds);
        let split_dim_u = split_dim as usize;

        // split along this dimension
        let split_val = (max_bounds[split_dim_u] + min_bounds[split_dim_u]) * T::from(0.5).unwrap();
        let range = max_bounds[split_dim_u] - min_bounds[split_dim_u];
        let (left_points, right_points) = if range == T::from(0).unwrap() {
            // degenerate data, split in half and iterate
            build_points.split_at_mut(build_points.len() / 2)
        } else {
            // partition data around split_val on split_dim
            partition::partition(build_points, |index| {
                cloud[*index].get(split_dim) < split_val
            })
        };
        debug_assert_ne!(left_points.len(), 0);
        debug_assert_ne!(right_points.len(), 0);

        // add this split
        self.nodes.push(Node::new_split_node(split_dim, split_val));

        // recurse
        let left_child = self.build_nodes(cloud, left_points);
        debug_assert_eq!(left_child, pos + 1);
        let right_child = self.build_nodes(cloud, right_points);

        // write right child index and return
        self.nodes[pos].set_child_index(right_child as u32);
        pos
    }

    fn get_build_points_bounds(
        cloud: &[P],
        build_points: &[usize],
    ) -> (Vec<NotNan<T>>, Vec<NotNan<T>>) {
        let mut min_bounds = vec![NotNan::<T>::max_value(); P::DIM as usize];
        let mut max_bounds = vec![NotNan::<T>::min_value(); P::DIM as usize];
        for p_index in build_points {
            let p = &cloud[*p_index];
            for index in 0..P::DIM {
                let index_u = index as usize;
                min_bounds[index_u] = clamp_max(p.get(index), min_bounds[index_u]);
                max_bounds[index_u] = clamp_min(p.get(index), max_bounds[index_u]);
            }
        }
        (min_bounds, max_bounds)
    }

    fn max_delta_index(lower_bound: &[NotNan<T>], upper_bound: &[NotNan<T>]) -> u32 {
        lower_bound
            .iter()
            .zip(upper_bound.iter())
            .enumerate()
            .max_by_key(|(_, (l, u))| *u - *l)
            .unwrap()
            .0 as u32
    }

    fn externalise_neighbour(&self, neighbour: InternalNeighbour<T>) -> Neighbour<T, P> {
        let mut point = P::default();
        let base_index = neighbour.index * P::DIM;
        for i in 0..P::DIM {
            point.set(i, self.points[(base_index + i) as usize]);
        }
        Neighbour {
            point,
            dist2: neighbour.dist2,
            index: self.indices[neighbour.index as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use dummy_point::{cloud_random, rand_point, P2};
    use float_cmp::approx_eq;

    // helpers to create cloud
    fn cloud3() -> Vec<P2> {
        vec![P2::new(0., 0.), P2::new(-1., 3.), P2::new(2., -4.)]
    }

    // helper to compute the square distance between two points
    fn point_dist2<T: Scalar, P: Point<T>>(lhs: &P, rhs: &P) -> NotNan<T> {
        let mut dist2 = NotNan::<T>::zero();
        for index in 0..P::DIM {
            let diff = lhs.get(index) - rhs.get(index);
            dist2 += diff * diff;
        }
        dist2
    }

    // brute force search implementations
    fn brute_force_1nn(cloud: &[P2], query: &P2) -> Neighbour<f32, P2> {
        let mut best_dist2 = f32::infinity();
        let mut best_index = 0;
        for (index, point) in cloud.iter().enumerate() {
            let dist2 = point_dist2(point, query).into_inner();
            if dist2 < best_dist2 {
                best_dist2 = dist2;
                best_index = index;
            }
        }
        Neighbour {
            point: cloud[best_index],
            dist2: NotNan::new(best_dist2).unwrap(),
            index: best_index as u32,
        }
    }

    fn brute_force_knn<H: CandidateHeap<f32>>(
        cloud: &[P2],
        query: &P2,
        k: u32,
    ) -> Vec<Neighbour<f32, P2>> {
        let mut h = H::new_with_k(k);
        for (index, point) in cloud.iter().enumerate() {
            let dist2 = point_dist2(point, query);
            h.add(dist2, index as u32);
        }
        h.into_sorted_vec()
            .into_iter()
            .map(|n| {
                let index = n.index as usize;
                Neighbour {
                    point: cloud[index],
                    dist2: n.dist2,
                    index: n.index,
                }
            })
            .collect()
    }

    // tests themselves

    #[test]
    fn test_get_build_points_bounds() {
        let cloud = cloud3();
        let indices = vec![0, 1, 2];
        let bounds = KDTree::get_build_points_bounds(&cloud, &indices);
        assert_eq!(bounds.0, vec![-1., -4.]);
        assert_eq!(bounds.1, vec![2., 3.]);
    }

    #[test]
    fn test_max_delta_index() {
        let b = |x: f32, y: f32| {
            [
                NotNan::<f32>::new(x).unwrap(),
                NotNan::<f32>::new(y).unwrap(),
            ]
        };
        assert_eq!(
            KDTree::<f32, P2>::max_delta_index(&b(0., 0.), &b(0., 1.)),
            1
        );
        assert_eq!(
            KDTree::<f32, P2>::max_delta_index(&b(0., 0.), &b(-1., 1.)),
            1
        );
        assert_eq!(
            KDTree::<f32, P2>::max_delta_index(&b(0., 0.), &b(-1., -2.)),
            0
        );
    }

    #[test]
    fn test_new() {
        let cloud = cloud3();
        let tree = KDTree::new_with_bucket_size(&cloud, 2);
        dbg!(tree);
    }

    #[test]
    fn test_1nn_allow_self() {
        let mut touch_sum = 0;
        const PASS_COUNT: u32 = 20;
        const QUERY_COUNT: u32 = 100;
        const CLOUD_SIZE: u32 = 1000;
        const PARAMETERS: Parameters<f32> = Parameters {
            epsilon: 0.0,
            max_radius: f32::INFINITY,
            allow_self_match: true,
            sort_results: true,
        };
        for _ in 0..PASS_COUNT {
            let cloud = cloud_random(CLOUD_SIZE);
            let tree = KDTree::new(&cloud);
            for _ in 0..QUERY_COUNT {
                let query = rand_point();
                let mut touch_statistics = 0;

                // linear search
                let nns_lin = tree.knn_advanced(
                    1,
                    &query,
                    CandidateContainer::Linear,
                    &PARAMETERS,
                    Some(&mut touch_statistics),
                );
                assert_eq!(nns_lin.len(), 1);
                let nn_lin = &nns_lin[0];
                assert_eq!(nn_lin.point, cloud[nn_lin.index as usize]);
                touch_sum += touch_statistics;
                // binary
                let nns_bin =
                    tree.knn_advanced(1, &query, CandidateContainer::BinaryHeap, &PARAMETERS, None);
                assert_eq!(nns_bin.len(), 1);
                let nn_bin = &nns_bin[0];
                assert_eq!(nn_bin.point, cloud[nn_bin.index as usize]);
                // brute force
                let nn_bf = brute_force_1nn(&cloud, &query);
                assert_eq!(nn_bf.point, cloud[nn_bf.index as usize]);
                // assertion
                assert_eq!(
                    nn_bin.index, nn_bf.index,
                    "KDTree binary heap: mismatch indexes\nquery: {}\npoint {}, {}\nvs bf {}, {}",
                    query, nn_bin.dist2, nn_bin.point, nn_bf.dist2, nn_bf.point
                );
                assert_eq!(nn_lin.index, nn_bf.index, "\nKDTree linear heap: mismatch indexes\nquery: {}\npoint {}, {}\nvs bf {}, {}\n", query, nn_lin.dist2, nn_lin.point, nn_bf.dist2, nn_bf.point);
                assert!(approx_eq!(f32, *nn_lin.dist2, *nn_bf.dist2, ulps = 2));
                assert!(approx_eq!(f32, *nn_bin.dist2, *nn_bf.dist2, ulps = 2));
            }
        }
        let touch_pct = (touch_sum * 100) as f32 / (PASS_COUNT * QUERY_COUNT * CLOUD_SIZE) as f32;
        println!("Average tree point touched: {} %", touch_pct);
    }

    #[test]
    fn test_knn_allow_self() {
        const QUERY_COUNT: u32 = 100;
        const CLOUD_SIZE: u32 = 1000;
        const PARAMETERS: Parameters<f32> = Parameters {
            epsilon: 0.0,
            max_radius: f32::INFINITY,
            allow_self_match: true,
            sort_results: true,
        };
        let cloud = cloud_random(CLOUD_SIZE);
        let tree = KDTree::new(&cloud);
        for k in [1, 2, 3, 5, 7, 13] {
            for _ in 0..QUERY_COUNT {
                let query = rand_point();
                // brute force
                let nns_bf_lin = brute_force_knn::<Vec<InternalNeighbour<f32>>>(&cloud, &query, k);
                assert_eq!(nns_bf_lin.len(), k as usize);
                let nns_bf_bin =
                    brute_force_knn::<BinaryHeap<InternalNeighbour<f32>>>(&cloud, &query, k);
                assert_eq!(nns_bf_bin.len(), k as usize);
                // kd-tree
                #[cfg_attr(rustfmt, rustfmt_skip)]
                let nns_bin = tree.knn_advanced(
                    k, &query,
                    CandidateContainer::BinaryHeap,
                    &PARAMETERS,
                    None,
                );
                assert_eq!(nns_bin.len(), k as usize);
                #[cfg_attr(rustfmt, rustfmt_skip)]
                let nns_lin = tree.knn_advanced(
                    k, &query,
                    CandidateContainer::Linear,
                    &PARAMETERS,
                    None,
                );
                assert_eq!(nns_lin.len(), k as usize);
                // assertion
                for i in 0..k as usize {
                    // get neighbour
                    let nn_bf_lin = &nns_bf_lin[i];
                    let nn_bf_bin = &nns_bf_bin[i];
                    let nn_lin = &nns_lin[i];
                    let nn_bin = &nns_bin[i];
                    // ensure their point data are consistent with the cloud
                    assert_eq!(nn_bf_lin.point, cloud[nn_bf_lin.index as usize]);
                    assert_eq!(nn_bf_bin.point, cloud[nn_bf_bin.index as usize]);
                    assert_eq!(nn_lin.point, cloud[nn_lin.index as usize]);
                    assert_eq!(nn_bin.point, cloud[nn_bin.index as usize]);
                    // ensure their indices are consistent
                    assert_eq!(nn_bf_bin.index, nn_bf_lin.index, "BF binary heap: mismatch indexes at {} on {}\nquery: {}\n   bf bin {}, {}\nvs bf lin {}, {}\n", i, k, query, nn_bf_bin.dist2, nn_bf_bin.point, nn_bf_lin.dist2, nn_bf_lin.point);
                    assert_eq!(nn_lin.index, nn_bf_lin.index, "\nKDTree linear heap: mismatch indexes at {} on {}\nquery: {}\npoint {}, {}\nvs bf {}, {}\n", i, k, query, nn_lin.dist2, nn_lin.point, nn_bf_lin.dist2, nn_bf_lin.point);
                    assert_eq!(nn_bin.index, nn_bf_lin.index, "\nKDTree binary heap: mismatch indexes {} on {}\nquery: {}\npoint {}, {}\nvs bf {}, {}\n", i, k, query, nn_bin.dist2, nn_bin.point, nn_bf_lin.dist2, nn_bf_lin.point);
                    // ensure their dist2 are consistent
                    assert!(approx_eq!(
                        f32,
                        *nn_bf_bin.dist2,
                        *nn_bf_lin.dist2,
                        ulps = 2
                    ));
                    assert!(approx_eq!(f32, *nn_lin.dist2, *nn_bf_lin.dist2, ulps = 2));
                    assert!(approx_eq!(f32, *nn_bin.dist2, *nn_bf_lin.dist2, ulps = 2));
                }
            }
        }
    }
}
