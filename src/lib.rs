#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[macro_use]
extern crate serde;

#[cfg(test)]
extern crate maplit;

extern crate sorted_iter;
pub use sorted_iter::{SortedIterator, SortedPairIterator};

#[cfg(test)]
#[macro_use]
mod test_macros;

mod small_vec_builder;
mod binary_merge;
mod merge_state;

mod dedup;
mod iterators;

#[cfg(test)]
mod obey;

pub mod range_set;

pub use range_set::RangeSet;