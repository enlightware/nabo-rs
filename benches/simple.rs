use nabo::dummy_point::*;
use nabo::KDTree;

fn main() {
    const QUERY_COUNT: u32 = 20000;
    const CLOUD_SIZE: u32 = 1000000;
    let cloud = cloud_random(CLOUD_SIZE);
    let tree = KDTree::new(&cloud);
    let queries = (0..QUERY_COUNT).map(|_| rand_point()).collect::<Vec<_>>();
    for k in [1, 2, 3, 4, 6, 8, 11, 16, 24] {
        for query in &queries {
            tree.knn(k, query);
        }
    }
}
