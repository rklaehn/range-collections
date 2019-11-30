use crate::binary_merge::{EarlyOut, MergeOperation, MergeStateRead};
use crate::iterators::SliceIterator;
use crate::small_vec_builder::{InPlaceSmallVecBuilder, SmallVecIntoIter};
use smallvec::{Array, SmallVec};
use std::default::Default;
use std::fmt::Debug;

/// A typical write part for the merge state
pub(crate) trait MergeStateMut: MergeStateRead {
    /// Consume n elements of a
    fn advance_a(&mut self, n: usize, take: bool) -> EarlyOut;
    /// Consume n elements of b
    fn advance_b(&mut self, n: usize, take: bool) -> EarlyOut;
}

pub(crate) struct InPlaceMergeState<A: Array, B: Array> {
    pub a: InPlaceSmallVecBuilder<A>,
    pub b: SmallVecIntoIter<B>,
}

impl<A: Array, B: Array> InPlaceMergeState<A, B> {
    fn new(a: SmallVec<A>, b: SmallVec<B>) -> Self {
        Self {
            a: a.into(),
            b: SmallVecIntoIter::new(b),
        }
    }
    fn result(self) -> SmallVec<A> {
        self.a.into_vec()
    }
}

impl<'a, A: Array, B: Array> MergeStateRead for InPlaceMergeState<A, B> {
    type A = A::Item;
    type B = B::Item;
    fn a_slice(&self) -> &[A::Item] {
        &self.a.source_slice()
    }
    fn b_slice(&self) -> &[B::Item] {
        self.b.as_slice()
    }
}

impl<'a, A: Array> MergeStateMut for InPlaceMergeState<A, A> {
    fn advance_a(&mut self, n: usize, take: bool) -> EarlyOut {
        self.a.consume(n, take);
        Some(())
    }
    fn advance_b(&mut self, n: usize, take: bool) -> EarlyOut {
        if take {
            self.a.extend_from_iter(&mut self.b, n);
        } else {
            for _ in 0..n {
                let _ = self.b.next();
            }
        }
        Some(())
    }
}

impl<'a, A: Array, B: Array> InPlaceMergeState<A, B> {
    pub fn merge<O: MergeOperation<Self>>(a: &mut SmallVec<A>, b: SmallVec<B>, o: O) {
        let mut t: SmallVec<A> = Default::default();
        std::mem::swap(a, &mut t);
        let mut state = Self::new(t, b);
        o.merge(&mut state);
        *a = state.result();
    }
}

/// A merge state where we only track if elements have been produced, and abort as soon as the first element is produced
pub(crate) struct BoolOpMergeState<'a, A, B> {
    a: SliceIterator<'a, A>,
    b: SliceIterator<'a, B>,
    r: bool,
}

impl<'a, A: Debug, B: Debug> Debug for BoolOpMergeState<'a, A, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "a: {:?}, b: {:?} r: {}",
            self.a_slice(),
            self.b_slice(),
            self.r
        )
    }
}

impl<'a, A, B> BoolOpMergeState<'a, A, B> {
    pub fn new(a: &'a [A], b: &'a [B]) -> Self {
        Self {
            a: SliceIterator(a),
            b: SliceIterator(b),
            r: false,
        }
    }
}

impl<'a, A, B> BoolOpMergeState<'a, A, B> {
    pub fn merge<O: MergeOperation<Self>>(a: &'a [A], b: &'a [B], o: O) -> bool {
        let mut state = Self::new(a, b);
        o.merge(&mut state);
        state.r
    }
}

impl<'a, A, B> MergeStateRead for BoolOpMergeState<'a, A, B> {
    type A = A;
    type B = B;
    fn a_slice(&self) -> &[A] {
        self.a.as_slice()
    }
    fn b_slice(&self) -> &[B] {
        self.b.as_slice()
    }
}

impl<'a, A, B> MergeStateMut for BoolOpMergeState<'a, A, B> {
    fn advance_a(&mut self, n: usize, take: bool) -> EarlyOut {
        if take {
            self.r = true;
            None
        } else {
            self.a.drop_front(n);
            Some(())
        }
    }
    fn advance_b(&mut self, n: usize, take: bool) -> EarlyOut {
        if take {
            self.r = true;
            None
        } else {
            self.b.drop_front(n);
            Some(())
        }
    }
}

/// A merge state where we build into a new vector
pub(crate) struct SmallVecMergeState<'a, A, B, Arr: Array> {
    pub a: SliceIterator<'a, A>,
    pub b: SliceIterator<'a, B>,
    pub r: SmallVec<Arr>,
}

impl<'a, A: Debug, B: Debug, Arr: Array> Debug for SmallVecMergeState<'a, A, B, Arr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a: {:?}, b: {:?}", self.a_slice(), self.b_slice(),)
    }
}

impl<'a, A, B, Arr: Array> SmallVecMergeState<'a, A, B, Arr> {
    pub fn new(a: &'a [A], b: &'a [B], r: SmallVec<Arr>) -> Self {
        Self {
            a: SliceIterator(a),
            b: SliceIterator(b),
            r,
        }
    }

    pub fn into_vec(self) -> SmallVec<Arr> {
        self.r
    }

    pub fn merge<O: MergeOperation<Self>>(a: &'a [A], b: &'a [B], o: O) -> SmallVec<Arr> {
        let t: SmallVec<Arr> = SmallVec::new();
        let mut state = Self::new(a, b, t);
        o.merge(&mut state);
        state.into_vec()
    }
}

impl<'a, A, B, Arr: Array> MergeStateRead for SmallVecMergeState<'a, A, B, Arr> {
    type A = A;
    type B = B;
    fn a_slice(&self) -> &[A] {
        self.a.as_slice()
    }
    fn b_slice(&self) -> &[B] {
        self.b.as_slice()
    }
}

impl<'a, T: Clone, Arr: Array<Item = T>> MergeStateMut for SmallVecMergeState<'a, T, T, Arr> {
    fn advance_a(&mut self, n: usize, take: bool) -> EarlyOut {
        if take {
            self.r.reserve(n);
            for e in self.a.take_front(n).iter() {
                self.r.push(e.clone())
            }
        } else {
            self.a.drop_front(n);
        }
        Some(())
    }
    fn advance_b(&mut self, n: usize, take: bool) -> EarlyOut {
        if take {
            self.r.reserve(n);
            for e in self.b.take_front(n).iter() {
                self.r.push(e.clone())
            }
        } else {
            self.b.drop_front(n);
        }
        Some(())
    }
}
