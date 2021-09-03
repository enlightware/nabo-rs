use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nabo::dummy_point::*;
use nabo::CandidateContainer;
use nabo::KDTree;
use nabo::Parameters;

fn bench_candidate_container_types(c: &mut Criterion) {
    const QUERY_COUNT: u32 = 10000;
    const CLOUD_SIZE: u32 = 1000000;
    const PARAMETERS: Parameters<f32> = Parameters {
        epsilon: 0.0,
        max_radius: f32::INFINITY,
        allow_self_match: true,
        sort_results: true,
    };
    let cloud = cloud_random(CLOUD_SIZE);
    let tree = KDTree::new(&cloud);
    let queries = (0..QUERY_COUNT).map(|_| rand_point()).collect::<Vec<_>>();
    let mut group = c.benchmark_group("CandidateContainerType");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for k in [1, 2, 3, 4, 6, 8, 11, 16, 24, 37, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("BinaryHeap", k),
            &(k, &tree, &queries),
            |b, (k, tree, queries)| {
                b.iter(|| {
                    for query in *queries {
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        tree.knn_advanced(
                            *k, query,
                            CandidateContainer::BinaryHeap,
                            &PARAMETERS,
                            None,
                        );
                    }
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Linear", k),
            &(k, &tree, &queries),
            |b, (k, tree, queries)| {
                b.iter(|| {
                    for query in *queries {
                        #[cfg_attr(rustfmt, rustfmt_skip)]
                        tree.knn_advanced(
                            *k, query,
                            CandidateContainer::Linear,
                            &PARAMETERS,
                            None,
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_candidate_container_types);
criterion_main!(benches);
