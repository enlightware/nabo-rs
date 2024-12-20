#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use nabo::simple_point::*;
use nabo::KDTree;

fn main() {
    const QUERY_COUNT: u32 = 20000;
    const CLOUD_SIZE: u32 = 1000000;
    let cloud = random_point_cloud::<2>(CLOUD_SIZE);
    let tree = KDTree::new(&cloud);
    let queries = (0..QUERY_COUNT).map(|_| random_point()).collect::<Vec<_>>();
    for k in [1, 2, 3, 4, 6, 8, 11, 16, 24] {
        for query in &queries {
            tree.knn(k, query);
        }
    }
}
