use ordered_float::{Float, NotNan};
use std::cmp::Ordering;

use crate::infinite::HasInfinite;

/// An internal representation of neighbour, to avoid copying the point around
#[derive(Clone, Copy)]
pub(crate) struct InternalNeighbour<T: Float> {
    /// the index of this point
    pub(crate) index: u32,
    /// the distance to the point
    pub(crate) dist2: NotNan<T>,
}
impl<T: Float> Default for InternalNeighbour<T> {
    fn default() -> Self {
        InternalNeighbour {
            index: 0,
            dist2: NotNan::<T>::infinite(),
        }
    }
}

impl<T: Float> Eq for InternalNeighbour<T> {}

impl<T: Float> Ord for InternalNeighbour<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist2
            .cmp(&other.dist2)
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl<T: Float> PartialOrd for InternalNeighbour<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Float> PartialEq for InternalNeighbour<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: Float> HasInfinite for NotNan<T> {
    fn infinite() -> Self {
        NotNan::new(T::infinity()).unwrap()
    }
}
