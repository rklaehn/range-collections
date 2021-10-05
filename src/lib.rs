//! collections (set and maps) with ranges as keys, backed by SmallVec<T>
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[cfg(test)]
#[macro_use]
mod test_macros;

#[allow(dead_code)]
mod binary_merge;

#[allow(dead_code)]
mod merge_state;

#[allow(dead_code)]
mod small_vec_builder;

mod iterators;

#[cfg(test)]
mod obey;

pub mod range_set;

pub use range_set::{RangeSet, RangeSet2};
