use num_traits::Float;
use ordered_float::NotNan;

/// Parameters to be passed unchanged to internal recursive function
pub(crate) struct InternalParameters<T: Float> {
    pub(crate) max_error2: NotNan<T>,
    pub(crate) max_radius2: NotNan<T>,
    pub(crate) allow_self_match: bool,
}
