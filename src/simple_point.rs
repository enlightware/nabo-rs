//! A simple D-dimensional point type

use core::{
    fmt::Display,
    ops::{Add, Sub},
};
use ordered_float::NotNan;
#[cfg(any(feature = "rand", test))]
use rand::Rng;

use num_traits::Bounded;

use crate::Point;

/// A simple `f32` `D`-dimensional point type
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SimplePoint<const D: usize>(pub [NotNan<f32>; D]);
impl SimplePoint<2> {
    /// Creates a new point from (x,y).
    pub fn new2d(x: f32, y: f32) -> SimplePoint<2> {
        SimplePoint([NotNan::new(x).unwrap(), NotNan::new(y).unwrap()])
    }
    /// Creates a new point from (x,y,z).
    pub fn new3d(x: f32, y: f32, z: f32) -> SimplePoint<3> {
        SimplePoint([
            NotNan::new(x).unwrap(),
            NotNan::new(y).unwrap(),
            NotNan::new(z).unwrap(),
        ])
    }
}
impl<const D: usize> Default for SimplePoint<D> {
    fn default() -> SimplePoint<D> {
        SimplePoint([NotNan::new(0.0).unwrap(); D])
    }
}
impl<const D: usize> Bounded for SimplePoint<D> {
    fn min_value() -> SimplePoint<D> {
        SimplePoint([NotNan::<f32>::min_value(); D])
    }
    fn max_value() -> SimplePoint<D> {
        SimplePoint([NotNan::<f32>::max_value(); D])
    }
}
impl<const D: usize> Point<f32> for SimplePoint<D> {
    const DIM: u32 = D as u32;
    fn set(&mut self, index: u32, value: NotNan<f32>) {
        self.0[index as usize] = value;
    }
    fn get(&self, index: u32) -> NotNan<f32> {
        self.0[index as usize]
    }
}
impl<const D: usize> Add for SimplePoint<D> {
    type Output = SimplePoint<D>;

    fn add(self, rhs: SimplePoint<D>) -> Self::Output {
        SimplePoint(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}
impl<const D: usize> Sub for SimplePoint<D> {
    type Output = SimplePoint<D>;

    fn sub(self, rhs: SimplePoint<D>) -> Self::Output {
        SimplePoint(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}
impl<const D: usize> Display for SimplePoint<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{}, {}]", self.0[0], self.0[1])
    }
}

/// A simple 2-D point type
pub type P2 = SimplePoint<2>;

/// A simple 3-D point type
pub type P3 = SimplePoint<3>;

/// Creates a random point whose coordinate are in the interval [-100:100].
#[cfg(any(feature = "rand", test))]
pub fn random_point<const D: usize>() -> SimplePoint<D> {
    let mut rng = rand::thread_rng();
    SimplePoint(core::array::from_fn(|_| {
        NotNan::new(rng.gen_range(-100.0..100.0)).unwrap()
    }))
}

/// Creates a random cloud of count points using [random_point()] for each.
#[cfg(any(feature = "rand", test))]
pub fn random_point_cloud<const D: usize>(count: u32) -> alloc::vec::Vec<SimplePoint<D>> {
    (0..count).map(|_| random_point()).collect()
}
