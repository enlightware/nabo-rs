use std::marker::PhantomData;

use ordered_float::NotNan;

use crate::{Point, Scalar};

/// Either a split value or the index of the first point in the bucket
union SplitValOrBucketIndex<T: Scalar> {
    /// for split node, split value
    split_val: NotNan<T>,
    /// for leaf node, start index of bucket
    bucket_start_index: u32,
}

/// Creates the compound index containing the dimension and the index of the child or the bucket size
fn create_dim_child_bucket_size<T: Scalar, P: Point<T>>(
    dim: u32,
    child_index_or_bucket_size: u32,
) -> u32 {
    dim | (child_index_or_bucket_size << P::DIM_BIT_COUNT)
}

/// A node of the KD-Tree: either split or leaf
///
/// If split node, it holds a split dimension and the split value along this dimension,
/// and the index of its right child (its left child is its own index + 1).
/// If leaf node, it holds the start index of the bucket and the number of elements in the bucket.
pub(crate) struct Node<T: Scalar, P: Point<T>> {
    /// cut dimension for split nodes (dim_bit_count lsb), index of right node or number of points in bucket (rest).
    /// Note that left index is current + 1.
    dim_child_bucket_size: u32,
    /// Either a split value or the index of the first point in the bucket
    split_val_or_bucket_start_index: SplitValOrBucketIndex<T>,
    phantom: PhantomData<P>,
}
impl<T: Scalar, P: Point<T>> Node<T, P> {
    pub(crate) fn set_child_index(&mut self, child_index: u32) {
        self.dim_child_bucket_size |= child_index << P::DIM_BIT_COUNT;
    }
    pub(crate) fn new_split_node(split_dim: u32, split_val: NotNan<T>) -> Self {
        Node {
            dim_child_bucket_size: split_dim,
            split_val_or_bucket_start_index: SplitValOrBucketIndex { split_val },
            phantom: PhantomData,
        }
    }
    pub(crate) fn new_leaf_node(bucket_start_index: u32, bucket_size: u32) -> Self {
        Node {
            dim_child_bucket_size: create_dim_child_bucket_size::<T, P>(P::DIM, bucket_size),
            split_val_or_bucket_start_index: SplitValOrBucketIndex { bucket_start_index },
            phantom: PhantomData,
        }
    }
    /// Depending on the type of node (split or leaf), calls split_cb or leaf_cb with ctx as first argument
    #[inline]
    pub(crate) fn dispatch_on_type<Fl, Fs, Ctx, R>(&self, ctx: Ctx, split_cb: Fs, leaf_cb: Fl) -> R
    where
        Fl: FnOnce(Ctx, u32, u32) -> R, // ctx, bucket_start_index, bucket_size
        Fs: FnOnce(Ctx, u32, NotNan<T>, u32) -> R, // ctx, split_dim, split_val, right_child
    {
        // SAFETY: interpretation of cut_val_or_bucket_start_index is defined by the
        // P::DIM_MASK bits of self.dim_child_bucket_size
        // If they have value P::DIM this is a leaf node, otherwise it is a split
        // node and they specify the split axis.
        if self.dim_child_bucket_size & P::DIM_MASK == P::DIM {
            // leaf node
            let bucket_start_index =
                unsafe { self.split_val_or_bucket_start_index.bucket_start_index };
            let bucket_size = self.dim_child_bucket_size >> P::DIM_BIT_COUNT;
            leaf_cb(ctx, bucket_start_index, bucket_size)
        } else {
            // split node
            let split_val = unsafe { self.split_val_or_bucket_start_index.split_val };
            let split_dim = self.dim_child_bucket_size & P::DIM_MASK;
            let right_child = self.dim_child_bucket_size >> P::DIM_BIT_COUNT;
            split_cb(ctx, split_dim, split_val, right_child)
        }
    }
}
impl<T: Scalar, P: Point<T>> std::fmt::Debug for Node<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.dispatch_on_type(
            f,
            |f, split_dim, split_val, right_child| {
                f.debug_struct("Node(split)")
                    .field("split_dim", &split_dim)
                    .field("split_val", &split_val)
                    .field("right_child", &right_child)
                    .finish()
            },
            |f, bucket_start_index, bucket_size| {
                f.debug_struct("Node(leaf)")
                    .field("bucket_size", &bucket_size)
                    .field("bucket_start_index", &bucket_start_index)
                    .finish()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::dummy_point::P2;
    use crate::*;

    #[test]
    fn sizes() {
        dbg!(std::mem::size_of::<Node<f32, P2>>());
    }

    #[test]
    fn dim_bit_count() {
        let d: u32 = 4;
        let dim_bit_count = 32 - d.leading_zeros();
        assert_eq!(dim_bit_count, 3);
    }
}
