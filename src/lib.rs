#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[cfg(test)]
extern crate maplit;

#[cfg(test)]
#[macro_use]
mod test_macros;

mod small_vec_builder;
mod binary_merge;
mod merge_state;

mod iterators;

#[cfg(test)]
mod obey;

pub mod range_set;

pub use range_set::RangeSet;