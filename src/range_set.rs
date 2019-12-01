//! # About
//!
//! A set of non-overlapping ranges
//!
//! ```
//! # use vec_collections::RangeSet;
//! let mut a: RangeSet<i32> = RangeSet::from(10..);
//! let b: RangeSet<i32> = RangeSet::from(1..5);
//!
//! a |= b;
//! let r = !a;
//! ```
//!
//! A data structure to represent a set of non-overlapping ranges of element type `T: Ord`. It uses a `SmallVec<T>`
//! of sorted boundaries internally.
//!
//! It can represent not just finite ranges but also half-open and open ranges. Because it can represent infinite
//! ranges, it can also represent the set of all elements, and therefore all boolean operations including negation.
//!
//! It does not put any constraints on the element type for requriring an `Ord` instance. However, since it internally
//! uses an encoding similar to [std::ops::Range](https://doc.rust-lang.org/std/ops/struct.Range.html) with upper excluded
//! ranges, it can only represent single values for types that have a defined successor, such as integers. Adjacent ranges
//! will be merged.
//!
//! It provides very fast operations for set operations (&, |, ^) as well as for intersection tests (is_disjoint, is_subset).
//!
//! In addition to the fast set operations that produce a new range set, it also supports the equivalent in-place operations.
//!
//! # Complexity
//!
//! Complexity is given separately for the number of comparisons and the number of copies, since sometimes you have
//! a comparison operation that is basically free (any of the primitive types), whereas sometimes you have a comparison
//! operation that is many orders of magnitude more expensive than a copy (long strings, arbitrary precision integers, ...)
//!
//! ## Number of comparisons
//!
//! |operation    | best      | worst     | remark
//! |-------------|-----------|-----------|--------
//! |negation     | 1         | 1         |
//! |union        | O(log(N)) | O(N)      | binary merge
//! |intersection | O(log(N)) | O(N)      | binary merge
//! |difference   | O(log(N)) | O(N)      | binary merge
//! |xor          | O(log(N)) | O(N)      | binary merge
//! |membership   | O(log(N)) | O(log(N)) | binary search
//! |is_disjoint  | O(log(N)) | O(N)      | binary merge with cutoff
//! |is_subset    | O(log(N)) | O(N)      | binary merge with cutoff
//!
//! ## Number of copies
//!
//! For creating new sets, obviously there needs to be at least one copy for each element of the result set, so the
//! complexity is always O(N). For in-place operations it gets more interesting. In case the number of elements of
//! the result being identical to the number of existing elements, there will be no copies and no allocations.
//!
//! E.g. if the result just has some of the ranges of the left hand side extended or truncated, but the same number of boundaries,
//! there will be no allocations and no copies except for the changed boundaries themselves.
//!
//! If the result has fewer boundaries than then lhs, there will be some copying but no allocations. Only if the result
//! is larger than the capacity of the underlying vector of the lhs will there be allocations.
//!
//! |operation    | best      | worst     |
//! |-------------|-----------|-----------|
//! |negation     | 1         | 1         |
//! |union        | 1         | O(N)      |
//! |intersection | 1         | O(N)      |
//! |difference   | 1         | O(N)      |
//! |xor          | 1         | O(N)      |
//!
//! # Testing
//!
//! Testing is done by some simple smoke tests as well as quickcheck tests of the algebraic properties of the boolean operations.
use crate::binary_merge::{EarlyOut, MergeOperation, MergeStateRead};
use crate::merge_state::{BoolOpMergeState, InPlaceMergeState, MergeStateMut, SmallVecMergeState};
use smallvec::{Array, SmallVec};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Bound;
use std::ops::Bound::*;
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Range, RangeFrom, RangeTo,
    Sub, SubAssign,
};

#[derive(Clone, PartialEq, Eq)]
pub struct RangeSet<T, A: Array<Item = T> = [T; 2]> {
    below_all: bool,
    boundaries: SmallVec<A>,
}

impl<T: Debug, A: Array<Item = T>> Debug for RangeSet<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RangeSet{{")?;
        for (i, (l, u)) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match (l, u) {
                (Unbounded, Unbounded) => write!(f, ".."),
                (Unbounded, Excluded(b)) => write!(f, "..{:?}", b),
                (Included(a), Unbounded) => write!(f, "{:?}..", a),
                (Included(a), Excluded(b)) => write!(f, "{:?}..{:?}", a, b),
                _ => write!(f, ""),
            }?;
        }
        write!(f, "}}")
    }
}

pub struct Iter<'a, T>(bool, &'a [T]);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Bound<&'a T>, Bound<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        let (ul, bounds) = (self.0, self.1);
        if !bounds.is_empty() || ul {
            Some(if ul {
                self.0 = false;
                match bounds.split_first() {
                    None => (Unbounded, Unbounded),
                    Some((b, bs)) => {
                        self.1 = bs;
                        (Unbounded, Excluded(b))
                    }
                }
            } else if bounds.len() == 1 {
                self.1 = &bounds[1..];
                (Included(&bounds[0]), Unbounded)
            } else {
                self.1 = &bounds[2..];
                (Included(&bounds[0]), Excluded(&bounds[1]))
            })
        } else {
            None
        }
    }
}

impl<T, A: Array<Item = T>> RangeSet<T, A> {
    /// note that this is private since it does not check the invariants!
    fn new(below_all: bool, boundaries: SmallVec<A>) -> Self {
        RangeSet {
            below_all,
            boundaries,
        }
    }
    fn iter(&self) -> Iter<T> {
        Iter(self.below_all, self.boundaries.as_slice())
    }
    fn from_range_until(a: T) -> Self {
        let mut t = SmallVec::new();
        t.push(a);
        Self::new(true, t)
    }
    fn from_range_from(a: T) -> Self {
        let mut t = SmallVec::new();
        t.push(a);
        Self::new(false, t)
    }
    pub fn empty() -> Self {
        Self::new(false, SmallVec::new())
    }
    pub fn all() -> Self {
        Self::new(true, SmallVec::new())
    }
    pub fn constant(value: bool) -> Self {
        Self::new(value, SmallVec::new())
    }
    pub fn is_empty(&self) -> bool {
        !self.below_all && self.boundaries.is_empty()
    }
    pub fn is_all(&self) -> bool {
        self.below_all && self.boundaries.is_empty()
    }
}

impl<T: Ord, A: Array<Item = T>> RangeSet<T, A> {
    fn from_range(a: Range<T>) -> Self {
        if a.start < a.end {
            let mut t = SmallVec::new();
            t.push(a.start);
            t.push(a.end);
            Self::new(false, t)
        } else {
            Self::empty()
        }
    }

    pub fn is_disjoint(&self, that: &Self) -> bool {
        !RangeSetBoolOpMergeState::merge(
            self.below_all,
            self.boundaries.as_slice(),
            that.below_all,
            that.boundaries.as_slice(),
            IntersectionOp,
        )
    }

    pub fn is_superset(&self, that: &Self) -> bool {
        !RangeSetBoolOpMergeState::merge(
            that.below_all,
            that.boundaries.as_slice(),
            self.below_all,
            self.boundaries.as_slice(),
            DiffOp,
        )
    }

    pub fn is_subset(&self, that: &Self) -> bool {
        !RangeSetBoolOpMergeState::merge(
            self.below_all,
            self.boundaries.as_slice(),
            that.below_all,
            that.boundaries.as_slice(),
            DiffOp,
        )
    }

    pub fn contains(&self, value: &T) -> bool {
        match self.boundaries.binary_search(value) {
            Ok(index) => self.below_all ^ !is_odd(index),
            Err(index) => self.below_all ^ is_odd(index),
        }
    }
}

impl<T: Ord, A: Array<Item = T>> From<Range<T>> for RangeSet<T, A> {
    fn from(value: Range<T>) -> Self {
        Self::from_range(value)
    }
}

impl<T: Ord, A: Array<Item = T>> From<RangeFrom<T>> for RangeSet<T, A> {
    fn from(value: RangeFrom<T>) -> Self {
        Self::from_range_from(value.start)
    }
}

impl<T: Ord, A: Array<Item = T>> From<RangeTo<T>> for RangeSet<T, A> {
    fn from(value: RangeTo<T>) -> Self {
        Self::from_range_until(value.end)
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> BitAnd for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitand(self, that: Self) -> Self::Output {
        Self::Output::new(
            self.below_all & that.below_all,
            VecMergeState::merge(
                self.below_all,
                self.boundaries.as_slice(),
                that.below_all,
                that.boundaries.as_slice(),
                IntersectionOp,
            ),
        )
    }
}

impl<T: Ord, A: Array<Item = T>> BitAndAssign for RangeSet<T, A> {
    fn bitand_assign(&mut self, that: Self) {
        RangeSetInPlaceMergeState::merge(
            &mut self.boundaries,
            self.below_all,
            that.boundaries,
            that.below_all,
            IntersectionOp,
        );
        self.below_all &= that.below_all;
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> BitOr for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitor(self, that: Self) -> Self::Output {
        Self::Output::new(
            self.below_all | that.below_all,
            VecMergeState::merge(
                self.below_all,
                self.boundaries.as_slice(),
                that.below_all,
                that.boundaries.as_slice(),
                UnionOp,
            ),
        )
    }
}

impl<T: Ord, A: Array<Item = T>> BitOrAssign for RangeSet<T, A> {
    fn bitor_assign(&mut self, that: Self) {
        RangeSetInPlaceMergeState::merge(
            &mut self.boundaries,
            self.below_all,
            that.boundaries,
            that.below_all,
            UnionOp,
        );
        self.below_all |= that.below_all;
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> BitXor for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitxor(self, that: Self) -> Self::Output {
        Self::Output::new(
            self.below_all ^ that.below_all,
            VecMergeState::merge(
                self.below_all,
                self.boundaries.as_slice(),
                that.below_all,
                that.boundaries.as_slice(),
                XorOp,
            ),
        )
    }
}

impl<T: Ord, A: Array<Item = T>> BitXorAssign for RangeSet<T, A> {
    fn bitxor_assign(&mut self, that: Self) {
        RangeSetInPlaceMergeState::merge(
            &mut self.boundaries,
            self.below_all,
            that.boundaries,
            that.below_all,
            XorOp,
        );
        self.below_all ^= that.below_all;
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> Sub for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn sub(self, that: Self) -> Self::Output {
        Self::Output::new(
            self.below_all & !that.below_all,
            VecMergeState::merge(
                self.below_all,
                self.boundaries.as_slice(),
                that.below_all,
                that.boundaries.as_slice(),
                DiffOp,
            ),
        )
    }
}

impl<T: Ord, A: Array<Item = T>> SubAssign for RangeSet<T, A> {
    fn sub_assign(&mut self, that: Self) {
        RangeSetInPlaceMergeState::merge(
            &mut self.boundaries,
            self.below_all,
            that.boundaries,
            that.below_all,
            DiffOp,
        );
        self.below_all &= !that.below_all;
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> Not for RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn not(self) -> Self::Output {
        Self::new(!self.below_all, self.boundaries)
    }
}

impl<T: Ord + Clone, A: Array<Item = T>> Not for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn not(self) -> Self::Output {
        Self::Output::new(!self.below_all, self.boundaries.clone())
    }
}

fn is_odd(x: usize) -> bool {
    (x & 1) != 0
}

trait RangeSetMergeState: MergeStateRead + MergeStateMut {
    /// current state of a.
    fn ac(&self) -> bool;
    /// current state of b
    fn bc(&self) -> bool;
}

struct RangeSetBoolOpMergeState<'a, T> {
    inner: BoolOpMergeState<'a, T, T>,
    ac: bool,
    bc: bool,
}

impl<'a, T> RangeSetBoolOpMergeState<'a, T> {
    fn merge<O: MergeOperation<Self>>(a0: bool, a: &'a [T], b0: bool, b: &'a [T], o: O) -> bool {
        let mut state = Self {
            ac: a0,
            bc: b0,
            inner: BoolOpMergeState::new(a, b),
        };
        o.merge(&mut state);
        state.inner.result()
    }
}

impl<'a, T> MergeStateMut for RangeSetBoolOpMergeState<'a, T> {
    fn advance_a(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.ac ^= is_odd(n);
        self.inner.advance_a(n, copy)
    }
    fn advance_b(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.bc ^= is_odd(n);
        self.inner.advance_b(n, copy)
    }
}

impl<'a, T> RangeSetMergeState for RangeSetBoolOpMergeState<'a, T> {
    fn ac(&self) -> bool {
        self.ac
    }
    fn bc(&self) -> bool {
        self.bc
    }
}

impl<'a, T> MergeStateRead for RangeSetBoolOpMergeState<'a, T> {
    type A = T;
    type B = T;
    fn a_slice(&self) -> &[T] {
        self.inner.a_slice()
    }
    fn b_slice(&self) -> &[T] {
        self.inner.b_slice()
    }
}

struct VecMergeState<'a, T, A: Array> {
    inner: SmallVecMergeState<'a, T, T, A>,
    ac: bool,
    bc: bool,
}

impl<'a, T: Clone, A: Array<Item = T>> VecMergeState<'a, T, A> {
    fn merge<O: MergeOperation<Self>>(
        a0: bool,
        a: &'a [T],
        b0: bool,
        b: &'a [T],
        o: O,
    ) -> SmallVec<A> {
        let mut state = Self {
            ac: a0,
            bc: b0,
            inner: SmallVecMergeState::new(a, b, SmallVec::new()),
        };
        o.merge(&mut state);
        state.inner.result()
    }
}

impl<'a, T: Clone, A: Array<Item = T>> MergeStateMut for VecMergeState<'a, T, A> {
    fn advance_a(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.ac ^= is_odd(n);
        self.inner.advance_a(n, copy)
    }
    fn advance_b(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.bc ^= is_odd(n);
        self.inner.advance_b(n, copy)
    }
}

impl<'a, T: Clone, A: Array<Item = T>> RangeSetMergeState for VecMergeState<'a, T, A> {
    fn ac(&self) -> bool {
        self.ac
    }
    fn bc(&self) -> bool {
        self.bc
    }
}

impl<'a, T, A: Array<Item = T>> MergeStateRead for VecMergeState<'a, T, A> {
    type A = T;
    type B = T;
    fn a_slice(&self) -> &[T] {
        self.inner.a_slice()
    }
    fn b_slice(&self) -> &[T] {
        self.inner.b_slice()
    }
}

struct RangeSetInPlaceMergeState<A: Array> {
    inner: InPlaceMergeState<A, A>,
    ac: bool,
    bc: bool,
}

impl<T, A: Array<Item = T>> RangeSetInPlaceMergeState<A> {
    pub fn merge<O: MergeOperation<Self>>(
        a: &mut SmallVec<A>,
        a0: bool,
        b: SmallVec<A>,
        b0: bool,
        o: O,
    ) {
        let mut t: SmallVec<A> = Default::default();
        std::mem::swap(a, &mut t);
        let mut state = Self {
            ac: a0,
            bc: b0,
            inner: InPlaceMergeState::new(t, b),
        };
        o.merge(&mut state);
        *a = state.inner.result();
    }
}

impl<'a, T, A: Array<Item = T>> MergeStateRead for RangeSetInPlaceMergeState<A> {
    type A = T;
    type B = T;
    fn a_slice(&self) -> &[T] {
        self.inner.a_slice()
    }
    fn b_slice(&self) -> &[T] {
        self.inner.b_slice()
    }
}

impl<'a, T, A: Array<Item = T>> MergeStateMut for RangeSetInPlaceMergeState<A> {
    fn advance_a(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.ac ^= is_odd(n);
        self.inner.advance_a(n, copy)
    }
    fn advance_b(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.bc ^= is_odd(n);
        self.inner.advance_b(n, copy)
    }
}

impl<A: Array> RangeSetMergeState for RangeSetInPlaceMergeState<A> {
    fn ac(&self) -> bool {
        self.ac
    }
    fn bc(&self) -> bool {
        self.bc
    }
}

struct UnionOp;
struct IntersectionOp;
struct XorOp;
struct DiffOp;

impl<'a, T: Ord, M: RangeSetMergeState<A = T, B = T>> MergeOperation<M> for UnionOp {
    fn from_a(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_a(n, !m.bc())
    }
    fn from_b(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_b(n, !m.ac())
    }
    fn collision(&self, m: &mut M) -> EarlyOut {
        m.advance_both(m.ac() == m.bc())
    }
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

impl<'a, T: Ord, M: RangeSetMergeState<A = T, B = T>> MergeOperation<M> for IntersectionOp {
    fn from_a(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_a(n, m.bc())
    }
    fn from_b(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_b(n, m.ac())
    }
    fn collision(&self, m: &mut M) -> EarlyOut {
        m.advance_both(m.ac() == m.bc())
    }
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

impl<'a, T: Ord, M: RangeSetMergeState<A = T, B = T>> MergeOperation<M> for DiffOp {
    fn from_a(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_a(n, !m.bc())
    }
    fn from_b(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_b(n, m.ac())
    }
    fn collision(&self, m: &mut M) -> EarlyOut {
        m.advance_both(m.ac() == !m.bc())
    }
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

impl<'a, T: Ord, M: RangeSetMergeState<A = T, B = T>> MergeOperation<M> for XorOp {
    fn from_a(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_a(n, true)
    }
    fn from_b(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_b(n, true)
    }
    fn collision(&self, m: &mut M) -> EarlyOut {
        m.advance_both(false)
    }
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obey::*;
    use num_traits::PrimInt;
    use quickcheck::*;
    use std::collections::BTreeSet;
    use std::ops::RangeBounds;

    impl<T: Ord + Clone, A: Array<Item = T>> RangeSet<T, A> {
        fn from_range_bounds<R: RangeBounds<T>>(r: R) -> std::result::Result<Self, ()> {
            match (r.start_bound(), r.end_bound()) {
                (Bound::Unbounded, Bound::Unbounded) => Ok(Self::all()),
                (Bound::Unbounded, Bound::Excluded(b)) => Ok(Self::from_range_until(b.clone())),
                (Bound::Included(a), Bound::Unbounded) => Ok(Self::from_range_from(a.clone())),
                (Bound::Included(a), Bound::Excluded(b)) => Ok(Self::from_range(Range {
                    start: a.clone(),
                    end: b.clone(),
                })),
                _ => Err(()),
            }
        }
    }

    impl<T: Arbitrary + Ord, A: Array<Item = T> + Clone + 'static> Arbitrary for RangeSet<T, A> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let mut boundaries: Vec<T> = Arbitrary::arbitrary(g);
            let below_all: bool = Arbitrary::arbitrary(g);
            boundaries.truncate(2);
            boundaries.sort();
            boundaries.dedup();
            Self::new(below_all, boundaries.into())
        }
    }

    /// A range set can be seen as a set of elements, even though it does not actually contain the elements
    impl<E: PrimInt, A: Array<Item = E>> TestSamples<E, bool> for RangeSet<E, A> {
        fn samples(&self, res: &mut BTreeSet<E>) {
            res.insert(E::min_value());
            for x in self.boundaries.iter().cloned() {
                res.insert(x - E::one());
                res.insert(x);
                res.insert(x + E::one());
            }
            res.insert(E::max_value());
        }

        fn at(&self, elem: E) -> bool {
            self.contains(&elem)
        }
    }
    type Test = RangeSet<i64, [i64; 4]>;

    #[test]
    fn smoke_test() {
        let x: Test = Test::from(0..10);
        println!(
            "{:?} {:?} {:?} {:?} {:?}",
            x,
            x.contains(&0),
            x.contains(&1),
            x.contains(&9),
            x.contains(&10)
        );

        let y: Test = Test::from(..10);
        let z: Test = Test::from(20..);

        let r: Test = x.bitor(&z);

        println!("{:?} {:?} {:?} {:?}", x, y, z, r);

        let r2: Test = x.bitand(&y);
        let r3: Test = x.bitxor(&y);
        let r4 = y.is_disjoint(&z);
        let r5 = y.bitand(&z);

        println!("{:?}", r2);
        println!("{:?}", r3);
        println!("{:?} {:?}", r4, r5);
    }

    #[quickcheck]
    fn ranges_consistent(a: Test) -> bool {
        let mut b = Test::empty();
        for e in a.iter() {
            b |= Test::from_range_bounds(e).unwrap();
        }
        a == b
    }

    #[quickcheck]
    fn is_disjoint_sample(a: Test, b: Test) -> bool {
        binary_property_test(&a, &b, a.is_disjoint(&b), |a, b| !(a & b))
    }

    #[quickcheck]
    fn is_subset_sample(a: Test, b: Test) -> bool {
        binary_property_test(&a, &b, a.is_subset(&b), |a, b| !a | b)
    }

    #[quickcheck]
    fn negation_check(a: RangeSet<i64>) -> bool {
        unary_element_test(&a, !a.clone(), |x| !x)
    }

    #[quickcheck]
    fn union_check(a: RangeSet<i64>, b: RangeSet<i64>) -> bool {
        binary_element_test(&a, &b, &a | &b, |a, b| a | b)
    }

    #[quickcheck]
    fn intersection_check(a: RangeSet<i64>, b: RangeSet<i64>) -> bool {
        binary_element_test(&a, &b, &a & &b, |a, b| a & b)
    }

    #[quickcheck]
    fn xor_check(a: RangeSet<i64>, b: RangeSet<i64>) -> bool {
        binary_element_test(&a, &b, &a ^ &b, |a, b| a ^ b)
    }

    #[quickcheck]
    fn difference_check(a: RangeSet<i64>, b: RangeSet<i64>) -> bool {
        binary_element_test(&a, &b, &a - &b, |a, b| a & !b)
    }

    bitop_assign_consistent!(Test);
    bitop_symmetry!(Test);
    bitop_empty!(Test);
    bitop_sub_not_all!(Test);
    set_predicate_consistent!(Test);
}
