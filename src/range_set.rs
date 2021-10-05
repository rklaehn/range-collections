// #![deny(missing_docs)]

//! A set of non-overlapping ranges
use crate::binary_merge::{EarlyOut, MergeOperation, MergeStateRead};
use crate::merge_state::{
    BoolOpMergeState, InPlaceMergeState, InPlaceMergeStateRef, MergeStateMut, SmallVecMergeState,
};
use core::cmp::Ordering;
use core::fmt;
use core::fmt::Debug;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound, Bound::*, Not, Range,
    RangeFrom, RangeTo, Sub, SubAssign,
};
use num_traits::{Bounded, PrimInt};
#[cfg(feature = "serde")]
use serde::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    ser::{Serialize, SerializeSeq, Serializer},
};
use smallvec::{Array, SmallVec};
#[cfg(feature = "serde")]
use std::marker::PhantomData;

/// # A set of non-overlapping ranges
///
/// ```
/// # use range_collections::RangeSet;
/// let mut a: RangeSet<i32> = RangeSet::from(10..);
/// let b: RangeSet<i32> = RangeSet::from(1..5);
///
/// a |= b;
/// let r = !a;
/// ```
///
/// A data structure to represent a set of non-overlapping ranges of element type `T: Ord`. It uses a `SmallVec<T>`
/// of sorted boundaries internally.
///
/// It can represent not just finite ranges but also ranges with unbounded start or end. Because it can represent
/// infinite ranges, it can also represent the set of all elements, and therefore all boolean operations including negation.
///
/// It does not put any constraints on the element type for requriring an `Ord` instance. Adjacent ranges will be merged.
///
/// It provides very fast operations for set operations (&, |, ^) as well as for intersection tests (is_disjoint, is_subset).
///
/// In addition to the fast set operations that produce a new range set, it also supports the equivalent in-place operations.
///
/// # Complexity
///
/// Complexity is given separately for the number of comparisons and the number of copies, since sometimes you have
/// a comparison operation that is basically free (any of the primitive types), whereas sometimes you have a comparison
/// operation that is many orders of magnitude more expensive than a copy (long strings, arbitrary precision integers, ...)
///
/// ## Number of comparisons
///
/// |operation    | best      | worst     | remark
/// |-------------|-----------|-----------|--------
/// |negation     | 1         | 1         |
/// |union        | O(log(N)) | O(N)      | binary merge
/// |intersection | O(log(N)) | O(N)      | binary merge
/// |difference   | O(log(N)) | O(N)      | binary merge
/// |xor          | O(log(N)) | O(N)      | binary merge
/// |membership   | O(log(N)) | O(log(N)) | binary search
/// |is_disjoint  | O(log(N)) | O(N)      | binary merge with cutoff
/// |is_subset    | O(log(N)) | O(N)      | binary merge with cutoff
///
/// ## Number of copies
///
/// For creating new sets, obviously there needs to be at least one copy for each element of the result set, so the
/// complexity is always O(N). For in-place operations it gets more interesting. In case the number of elements of
/// the result being identical to the number of existing elements, there will be no copies and no allocations.
///
/// E.g. if the result just has some of the ranges of the left hand side extended or truncated, but the same number of boundaries,
/// there will be no allocations and no copies except for the changed boundaries themselves.
///
/// If the result has fewer boundaries than then lhs, there will be some copying but no allocations. Only if the result
/// is larger than the capacity of the underlying vector of the lhs will there be allocations.
///
/// |operation    | best      | worst     |
/// |-------------|-----------|-----------|
/// |negation     | 1         | 1         |
/// |union        | 1         | O(N)      |
/// |intersection | 1         | O(N)      |
/// |difference   | 1         | O(N)      |
/// |xor          | 1         | O(N)      |
///
/// # Testing
///
/// Testing is done by some simple smoke tests as well as quickcheck tests of the algebraic properties of the boolean operations.
#[derive(Clone, PartialEq, Eq)]
pub struct RangeSet<T, A: Array<Item = T> = [T; 2]>(SmallVec<A>);

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

/// Iterator for the ranges in a range set
pub struct Iter<'a, T>(&'a [T]);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Bound<&'a T>, Bound<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        let bounds = self.0;
        if !bounds.is_empty() {
            Some(if bounds.len() == 1 {
                self.0 = &bounds[1..];
                (Included(&bounds[0]), Unbounded)
            } else {
                self.0 = &bounds[2..];
                (Included(&bounds[0]), Excluded(&bounds[1]))
            })
        } else {
            None
        }
    }
}

/// trait for types that have a smallest value
pub trait MinValue {
    /// the minimum value for this type
    fn min_value() -> Self;
}

impl<T: PrimInt> MinValue for T {
    fn min_value() -> Self {
        <T as Bounded>::min_value()
    }
}

impl<T, A: Array<Item = T>> RangeSet<T, A> {
    /// note that this is private since it does not check the invariants!
    fn new(boundaries: SmallVec<A>) -> Self {
        RangeSet(boundaries)
    }

    /// iterate over all ranges in this range set
    pub fn iter(&self) -> Iter<T> {
        Iter(self.0.as_slice())
    }
    /// boundaries in this range set
    pub fn boundaries(&self) -> &SmallVec<A> {
        &self.0
    }
    /// get the boundaries in this range set as a SmallVec
    pub fn into_inner(self) -> SmallVec<A> {
        self.0
    }
}

impl<T: MinValue + Ord, A: Array<Item = T>> RangeSet<T, A> {
    fn from_range_until(a: T) -> Self {
        let mut t = SmallVec::new();
        let min = T::min_value();
        if a > min {
            t.push(min);
            t.push(a);
        }
        Self::new(t)
    }
    fn from_range_from(a: T) -> Self {
        let mut t = SmallVec::new();
        t.push(a);
        Self::new(t)
    }
    /// the empty range set
    pub fn empty() -> Self {
        Self(SmallVec::new())
    }
    /// a range set containing all values
    pub fn all() -> Self {
        Self::from_range_from(T::min_value())
    }
    /// true if the range set is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// true if the range set contains all values
    pub fn is_all(&self) -> bool {
        self.0.len() == 1 && self.0[0] == T::min_value()
    }
}

impl<T: MinValue + Ord + Clone, A: Array<Item = T>> RangeSet<T, A> {
    pub fn intersection(&self, that: &Self) -> Self {
        Self::new(VecMergeState::merge(
            self.0.as_slice(),
            that.0.as_slice(),
            IntersectionOp,
        ))
    }
    pub fn intersection_with(&mut self, that: &Self) {
        InPlaceMergeStateRef::merge(&mut self.0, &that.0, IntersectionOp);
    }
    pub fn union(&self, that: &Self) -> Self {
        Self::new(VecMergeState::merge(
            self.0.as_slice(),
            that.0.as_slice(),
            UnionOp,
        ))
    }
    pub fn union_with(&mut self, that: &Self) {
        InPlaceMergeStateRef::merge(&mut self.0, &that.0, UnionOp);
    }
    pub fn difference(&self, that: &Self) -> Self {
        Self::new(VecMergeState::merge(
            self.0.as_slice(),
            that.0.as_slice(),
            DiffOp,
        ))
    }
    pub fn difference_with(&mut self, that: &Self) {
        InPlaceMergeStateRef::merge(&mut self.0, &that.0, DiffOp);
    }
    pub fn symmetric_difference(&self, that: &Self) -> Self {
        Self::new(VecMergeState::merge(
            self.0.as_slice(),
            that.0.as_slice(),
            XorOp,
        ))
    }
    pub fn symmetric_difference_with(&mut self, that: &Self) {
        InPlaceMergeStateRef::merge(&mut self.0, &that.0, XorOp);
    }
}

impl<T: MinValue + Ord, A: Array<Item = T>> From<bool> for RangeSet<T, A> {
    fn from(value: bool) -> Self {
        if value {
            Self::all()
        } else {
            Self::empty()
        }
    }
}

impl<T: Ord + MinValue, A: Array<Item = T>> RangeSet<T, A> {
    fn from_range(a: Range<T>) -> Self {
        if a.start < a.end {
            let mut t = SmallVec::new();
            t.push(a.start);
            t.push(a.end);
            Self::new(t)
        } else {
            Self::empty()
        }
    }

    /// true if this range set is disjoint from another range set
    pub fn is_disjoint(&self, that: &Self) -> bool {
        !RangeSetBoolOpMergeState::merge(self.0.as_slice(), that.0.as_slice(), IntersectionOp)
    }

    /// true if this range set is a superset of another range set
    ///
    /// A range set is considered to be a superset of itself
    pub fn is_superset(&self, that: &Self) -> bool {
        that.is_subset(self)
    }

    /// true if this range set is a subset of another range set
    ///
    /// A range set is considered to be a subset of itself
    pub fn is_subset(&self, that: &Self) -> bool {
        !RangeSetBoolOpMergeState::merge(self.0.as_slice(), that.0.as_slice(), DiffOp)
    }

    /// true if the value is contained in the range set
    pub fn contains(&self, value: &T) -> bool {
        match self.0.binary_search(value) {
            Ok(index) => !is_odd(index),
            Err(index) => is_odd(index),
        }
    }
}

impl<T: Ord + MinValue, A: Array<Item = T>> From<Range<T>> for RangeSet<T, A> {
    fn from(value: Range<T>) -> Self {
        Self::from_range(value)
    }
}

impl<T: Ord + MinValue, A: Array<Item = T>> From<RangeFrom<T>> for RangeSet<T, A> {
    fn from(value: RangeFrom<T>) -> Self {
        Self::from_range_from(value.start)
    }
}

impl<T: Ord + MinValue, A: Array<Item = T>> From<RangeTo<T>> for RangeSet<T, A> {
    fn from(value: RangeTo<T>) -> Self {
        Self::from_range_until(value.end)
    }
}

/// compute the intersection of this range set with another, producing a new range set
///
/// &forall; t &isin; T, r(t) = a(t) & b(t)
impl<T: Ord + MinValue + Clone, A: Array<Item = T>> BitAnd for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitand(self, that: Self) -> Self::Output {
        self.intersection(that)
    }
}

impl<T: Ord, A: Array<Item = T>> BitAndAssign for RangeSet<T, A> {
    fn bitand_assign(&mut self, that: Self) {
        InPlaceMergeState::merge(&mut self.0, that.0, IntersectionOp);
    }
}

// impl<T: Ord, A: Array<Item = T>> BitAnd for RangeSet<T, A> {
//     type Output = Self;
//     fn bitand(mut self, that: Self) -> Self::Output {
//         self &= that;
//         self
//     }
// }

/// compute the union of this range set with another, producing a new range set
///
/// &forall; t &isin; T, r(t) = a(t) | b(t)
impl<T: Ord + Clone + MinValue, A: Array<Item = T>> BitOr for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitor(self, that: Self) -> Self::Output {
        self.union(that)
    }
}

impl<T: Ord, A: Array<Item = T>> BitOrAssign for RangeSet<T, A> {
    fn bitor_assign(&mut self, that: Self) {
        InPlaceMergeState::merge(&mut self.0, that.0, UnionOp);
    }
}

// impl<T: Ord, A: Array<Item = T>> BitOr for RangeSet<T, A> {
//     type Output = Self;
//     fn bitor(mut self, that: Self) -> Self::Output {
//         self |= that;
//         self
//     }
// }

/// compute the exclusive or of this range set with another, producing a new range set
///
/// &forall; t &isin; T, r(t) = a(t) ^ b(t)
impl<T: Ord + Clone + MinValue, A: Array<Item = T>> BitXor for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn bitxor(self, that: Self) -> Self::Output {
        self.symmetric_difference(that)
    }
}

impl<T: Ord + MinValue, A: Array<Item = T>> BitXorAssign for RangeSet<T, A> {
    fn bitxor_assign(&mut self, that: Self) {
        InPlaceMergeState::merge(&mut self.0, that.0, XorOp);
    }
}

// impl<T: Ord, A: Array<Item = T>> BitXor for RangeSet<T, A> {
//     type Output = Self;
//     fn bitxor(mut self, that: Self) -> Self::Output {
//         self ^= that;
//         self
//     }
// }

/// compute the difference of this range set with another, producing a new range set
///
/// &forall; t &isin; T, r(t) = a(t) & !b(t)
impl<T: Ord + Clone + MinValue, A: Array<Item = T>> Sub for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn sub(self, that: Self) -> Self::Output {
        self.difference(that)
    }
}

impl<T: Ord, A: Array<Item = T>> SubAssign for RangeSet<T, A> {
    fn sub_assign(&mut self, that: Self) {
        InPlaceMergeState::merge(&mut self.0, that.0, DiffOp);
    }
}

// impl<T: Ord, A: Array<Item = T>> Sub for RangeSet<T, A> {
//     type Output = Self;
//     fn sub(mut self, that: Self) -> Self::Output {
//         self -= that;
//         self
//     }
// }

/// compute the negation of this range set
///
/// &forall; t &isin; T, r(t) = !a(t)
impl<T: Ord + Clone + MinValue, A: Array<Item = T>> Not for RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn not(mut self) -> Self::Output {
        match self.0.get(0) {
            Some(x) if *x == T::min_value() => {
                self.0.remove(0);
            }
            _ => {
                self.0.insert(0, T::min_value());
            }
        }
        self
    }
}

/// compute the negation of this range set
///
/// &forall; t &isin; T, r(t) = !a(t)
impl<T: Ord + Clone + MinValue, A: Array<Item = T>> Not for &RangeSet<T, A> {
    type Output = RangeSet<T, A>;
    fn not(self) -> Self::Output {
        self ^ &RangeSet::all()
    }
}

#[inline]
fn is_odd(x: usize) -> bool {
    (x & 1) != 0
}

struct RangeSetBoolOpMergeState<'a, T> {
    inner: BoolOpMergeState<'a, T, T>,
}

impl<'a, T> RangeSetBoolOpMergeState<'a, T> {
    fn merge<O: MergeOperation<Self>>(a: &'a [T], b: &'a [T], o: O) -> bool {
        let mut state = Self {
            inner: BoolOpMergeState::new(a, b),
        };
        o.merge(&mut state);
        state.inner.result()
    }
}

impl<'a, T> MergeStateMut for RangeSetBoolOpMergeState<'a, T> {
    fn advance_a(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.inner.advance_a(n, copy)
    }
    fn advance_b(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.inner.advance_b(n, copy)
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

    fn ac(&self) -> bool {
        self.inner.ac()
    }

    fn bc(&self) -> bool {
        self.inner.bc()
    }
}

struct VecMergeState<'a, T, A: Array> {
    inner: SmallVecMergeState<'a, T, T, A>,
}

impl<'a, T: Clone, A: Array<Item = T>> VecMergeState<'a, T, A> {
    fn merge<O: MergeOperation<Self>>(a: &'a [T], b: &'a [T], o: O) -> SmallVec<A> {
        let mut state = Self {
            inner: SmallVecMergeState::new(a, b, SmallVec::new()),
        };
        o.merge(&mut state);
        state.inner.result()
    }
}

impl<'a, T: Clone, A: Array<Item = T>> MergeStateMut for VecMergeState<'a, T, A> {
    fn advance_a(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.inner.advance_a(n, copy)
    }
    fn advance_b(&mut self, n: usize, copy: bool) -> EarlyOut {
        self.inner.advance_b(n, copy)
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
    fn ac(&self) -> bool {
        self.inner.ac()
    }
    fn bc(&self) -> bool {
        self.inner.bc()
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize, A: Array<Item = T>> Serialize for RangeSet<T, A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_seq(Some(self.0.len()))?;
        for x in self.0.iter() {
            map.serialize_element(x)?;
        }
        map.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de> + Ord, A: Array<Item = T>> Deserialize<'de> for RangeSet<T, A> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(RangeSetVisitor {
            phantom: PhantomData,
        })
    }
}

#[cfg(feature = "serde")]
struct RangeSetVisitor<T, A> {
    phantom: PhantomData<(T, A)>,
}

#[cfg(feature = "serde")]
impl<'de, T, A: Array<Item = T>> Visitor<'de> for RangeSetVisitor<T, A>
where
    T: Deserialize<'de> + Ord,
{
    type Value = RangeSet<T, A>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<B>(self, mut seq: B) -> Result<Self::Value, B::Error>
    where
        B: SeqAccess<'de>,
    {
        let len = seq.size_hint().unwrap_or(0);
        let mut boundaries: SmallVec<A> = SmallVec::with_capacity(len);
        while let Some(value) = seq.next_element::<A::Item>()? {
            boundaries.push(value);
        }
        boundaries.sort();
        boundaries.dedup();
        Ok(RangeSet(boundaries))
    }
}

#[cfg(feature = "rkyv")]
impl<T, A> rkyv::Archive for RangeSet<T, A>
where
    T: rkyv::Archive,
    A: Array<Item = T>,
{
    type Archived = ArchivedRangeSet<<T as rkyv::Archive>::Archived>;

    type Resolver = rkyv::vec::VecResolver;

    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        rkyv::vec::ArchivedVec::resolve_from_slice(
            self.0.as_slice(),
            pos,
            resolver,
            &mut (*out).0 as *mut rkyv::vec::ArchivedVec<<T as rkyv::Archive>::Archived>,
        );
    }
}

#[cfg(feature = "rkyv")]
impl<T, S, A> rkyv::Serialize<S> for RangeSet<T, A>
where
    T: rkyv::Archive + rkyv::Serialize<S>,
    S: rkyv::ser::ScratchSpace + rkyv::ser::Serializer,
    A: Array<Item = T>,
{
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        rkyv::vec::ArchivedVec::serialize_from_slice(self.0.as_ref(), serializer)
    }
}

#[cfg(feature = "rkyv")]
impl<T, A, D> rkyv::Deserialize<RangeSet<T, A>, D> for ArchivedRangeSet<T::Archived>
where
    T: rkyv::Archive,
    A: Array<Item = T>,
    D: rkyv::Fallible + ?Sized,
    [T::Archived]: rkyv::DeserializeUnsized<[T], D>,
{
    fn deserialize(&self, deserializer: &mut D) -> Result<RangeSet<T, A>, D::Error> {
        // todo: replace this with SmallVec once smallvec support for rkyv lands on crates.io
        let boundaries: Vec<T> = self.0.deserialize(deserializer)?;
        Ok(RangeSet(boundaries.into()))
    }
}

/// Archived version of a RangeSet
#[cfg(feature = "rkyv")]
#[derive(Debug)]
#[repr(transparent)]
pub struct ArchivedRangeSet<T>(rkyv::vec::ArchivedVec<T>);

/// Validation error for a range set
#[cfg(feature = "rkyv_validated")]
#[derive(Debug)]
pub enum ArchivedRangeSetError {
    /// error with the individual fields of the ArchivedRangeSet, e.g a NonZeroU64 with a value of 0
    ValueCheckError,
    /// boundaries were not properly ordered
    OrderCheckError,
}

#[cfg(feature = "rkyv_validated")]
impl std::error::Error for ArchivedRangeSetError {}

#[cfg(feature = "rkyv_validated")]
impl std::fmt::Display for ArchivedRangeSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(feature = "rkyv_validated")]
impl<C: ?Sized, T> bytecheck::CheckBytes<C> for ArchivedRangeSet<T>
where
    T: Ord,
    bool: bytecheck::CheckBytes<C>,
    rkyv::vec::ArchivedVec<T>: bytecheck::CheckBytes<C>,
{
    type Error = ArchivedRangeSetError;
    unsafe fn check_bytes<'a>(
        value: *const Self,
        context: &mut C,
    ) -> Result<&'a Self, Self::Error> {
        let boundaries = &(*value).0;
        rkyv::vec::ArchivedVec::<T>::check_bytes(boundaries, context)
            .map_err(|_| ArchivedRangeSetError::ValueCheckError)?;
        if !boundaries
            .iter()
            .zip(boundaries.iter().skip(1))
            .all(|(a, b)| a < b)
        {
            return Err(ArchivedRangeSetError::OrderCheckError);
        };
        Ok(&*value)
    }
}

struct UnionOp;
struct IntersectionOp;
struct XorOp;
struct DiffOp;

impl<'a, T: Ord, M: MergeStateMut<A = T, B = T>> MergeOperation<M> for UnionOp {
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

impl<'a, T: Ord, M: MergeStateMut<A = T, B = T>> MergeOperation<M> for IntersectionOp {
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

impl<'a, T: Ord, M: MergeStateMut<A = T, B = T>> MergeOperation<M> for DiffOp {
    fn from_a(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_a(n, !m.bc())
    }
    fn from_b(&self, m: &mut M, n: usize) -> EarlyOut {
        m.advance_b(n, m.ac())
    }
    fn collision(&self, m: &mut M) -> EarlyOut {
        m.advance_both(m.ac() != m.bc())
    }
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

impl<'a, T: Ord, M: MergeStateMut<A = T, B = T>> MergeOperation<M> for XorOp {
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

    impl<T: Ord + Clone + MinValue, A: Array<Item = T>> RangeSet<T, A> {
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
            boundaries.truncate(4);
            boundaries.sort();
            boundaries.dedup();
            Self::new(boundaries.into())
        }
    }

    /// A range set can be seen as a set of elements, even though it does not actually contain the elements
    impl<E: PrimInt, A: Array<Item = E>> TestSamples<E, bool> for RangeSet<E, A> {
        fn samples(&self, res: &mut BTreeSet<E>) {
            res.insert(E::min_value());
            for x in self.0.iter().cloned() {
                res.insert(x.saturating_sub(E::one()));
                res.insert(x);
                res.insert(x.saturating_add(E::one()));
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

        let r: Test = (&x).bitor(&z);

        println!("{:?} {:?} {:?} {:?}", x, y, z, r);

        let r2: Test = (&x).bitand(&y);
        let r3: Test = (&x).bitxor(&y);
        let r4 = (&y).is_disjoint(&z);
        let r5 = (&y).bitand(&z);

        println!("{:?}", r2);
        println!("{:?}", r3);
        println!("{:?} {:?}", r4, r5);
    }

    #[cfg(feature = "serde")]
    #[quickcheck]
    fn range_seq_serde(a: Test) -> bool {
        let bytes = serde_cbor::to_vec(&a).unwrap();
        let b: Test = serde_cbor::from_slice(&bytes).unwrap();
        a == b
    }

    #[cfg(feature = "rkyv")]
    #[quickcheck]
    fn range_seq_rkyv_unvalidated(a: Test) -> bool {
        use rkyv::*;
        use ser::Serializer;
        let mut serializer = ser::serializers::AllocSerializer::<256>::default();
        serializer.serialize_value(&a).unwrap();
        let bytes = serializer.into_serializer().into_inner();
        let archived = unsafe { rkyv::archived_root::<Test>(&bytes) };
        let deserialized: Test = archived.deserialize(&mut Infallible).unwrap();
        a == deserialized
    }

    #[cfg(feature = "rkyv_validated")]
    #[quickcheck]
    fn range_seq_rkyv_validated(a: Test) -> bool {
        use rkyv::*;
        use ser::Serializer;
        let mut serializer = ser::serializers::AllocSerializer::<256>::default();
        serializer.serialize_value(&a).unwrap();
        let bytes = serializer.into_serializer().into_inner();
        let archived = rkyv::check_archived_root::<Test>(&bytes).unwrap();
        let deserialized: Test = archived.deserialize(&mut Infallible).unwrap();
        a == deserialized
    }

    #[cfg(feature = "rkyv_validated")]
    #[test]
    fn range_seq_rkyv_errors() {
        use rkyv::*;
        use std::num::NonZeroU64;

        // deserialize a boolean value of 2, must produce an error!
        let mut bytes = AlignedVec::new();
        bytes.extend_from_slice(&hex::decode("000000000000000002000000").unwrap());
        assert!(rkyv::check_archived_root::<Test>(&bytes).is_err());

        // deserialize wrongly ordered range set, must produce an error
        let mut bytes = AlignedVec::new();
        bytes.extend_from_slice(
            &hex::decode("02000000000000000100000000000000f0ffffff0200000000000000").unwrap(),
        );
        assert!(rkyv::check_archived_root::<Test>(&bytes).is_err());

        // deserialize wrong value (0 for a NonZeroU64), must produce an error
        let mut bytes = AlignedVec::new();
        bytes.extend_from_slice(
            &hex::decode("00000000000000000100000000000000f0ffffff0200000000000000").unwrap(),
        );
        assert!(rkyv::check_archived_root::<RangeSet<NonZeroU64>>(&bytes).is_err());
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
        let res = binary_property_test(&a, &b, a.is_disjoint(&b), |a, b| !(a & b));
        if !res {
            println!("{:?} {:?} {:?}", a, b, a.is_disjoint(&b));
        }
        res
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
