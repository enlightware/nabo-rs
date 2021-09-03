use std::collections::BinaryHeap;

use ordered_float::NotNan;

use crate::internal_neighbour::InternalNeighbour;
use crate::Scalar;

/// Local trait for keeping candidates in a heap-like behaviour
pub(crate) trait CandidateHeap<T: Scalar> {
    fn new_with_k(k: u32) -> Self;
    fn add(&mut self, dist2: NotNan<T>, index: u32);
    fn furthest_dist2(&self) -> NotNan<T>;
    fn into_vec(self) -> Vec<InternalNeighbour<T>>;
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>>;
}

impl<T: Scalar> CandidateHeap<T> for BinaryHeap<InternalNeighbour<T>> {
    fn new_with_k(k: u32) -> Self {
        BinaryHeap::with_capacity(k as usize)
    }
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        let k = self.capacity();
        if self.len() < k {
            self.push(InternalNeighbour { index, dist2 });
        } else {
            let mut max_heap_value = self.peek_mut().unwrap();
            if dist2 < max_heap_value.dist2 {
                *max_heap_value = InternalNeighbour { index, dist2 };
            }
        }
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        if self.len() < self.capacity() {
            NotNan::new(T::infinity()).unwrap()
        } else {
            self.peek()
                .map_or(NotNan::new(T::infinity()).unwrap(), |n| n.dist2)
        }
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        BinaryHeap::into_vec(self)
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        BinaryHeap::into_sorted_vec(self)
    }
}

impl<T: Scalar> CandidateHeap<T> for Vec<InternalNeighbour<T>> {
    fn new_with_k(k: u32) -> Self {
        vec![
            InternalNeighbour {
                index: 0,
                dist2: NotNan::new(T::infinity()).unwrap()
            };
            k as usize
        ]
    }
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        if dist2 > self.furthest_dist2() {
            return;
        }
        let mut i = self.len() - 1;
        while i > 0 {
            if self[i - 1].dist2 > dist2 {
                self[i] = self[i - 1];
            } else {
                break;
            }
            i -= 1;
        }
        self[i].dist2 = dist2;
        self[i].index = index;
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        self[self.len() - 1].dist2
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        self
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        self
    }
}

impl<T: Scalar> CandidateHeap<T> for InternalNeighbour<T> {
    fn new_with_k(k: u32) -> Self {
        debug_assert_eq!(k, 1);
        InternalNeighbour {
            index: 0,
            dist2: NotNan::new(T::infinity()).unwrap(),
        }
    }
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        if dist2 < self.dist2 {
            self.dist2 = dist2;
            self.index = index;
        }
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        self.dist2
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        vec![self]
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        vec![self]
    }
}
