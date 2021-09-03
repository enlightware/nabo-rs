/// Local trait for providing infinity, working-around the Rust trait impl limitations
pub(crate) trait HasInfinite {
    fn infinite() -> Self;
}
