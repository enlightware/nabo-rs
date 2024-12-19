use core::cmp::Ordering;
use ordered_float::{FloatCore, NotNan};

use crate::infinite::HasInfinite;

/// An internal representation of neighbour, to avoid copying the point around
#[derive(Clone, Copy, Debug)]
pub(crate) struct InternalNeighbour<T: FloatCore> {
    /// the index of this point
    pub(crate) index: u32,
    /// the distance to the point
    pub(crate) dist2: NotNan<T>,
}
impl<T: FloatCore> Default for InternalNeighbour<T> {
    fn default() -> Self {
        InternalNeighbour {
            index: 0,
            dist2: NotNan::<T>::infinite(),
        }
    }
}

impl<T: FloatCore> Eq for InternalNeighbour<T> {}

impl<T: FloatCore> Ord for InternalNeighbour<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist2
            .cmp(&other.dist2)
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl<T: FloatCore> PartialOrd for InternalNeighbour<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: FloatCore> PartialEq for InternalNeighbour<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T: FloatCore> HasInfinite for NotNan<T> {
    fn infinite() -> Self {
        NotNan::new(T::infinity()).unwrap()
    }
}
