//! A data structure for in place modification of vecs.
// #![deny(warnings)]
#![deny(missing_docs)]
use smallvec::{Array, SmallVec};
use std::fmt::Debug;

/// builds a SmallVec out of itself
pub struct InPlaceSmallVecBuilder<A: Array> {
    /// the underlying vector, possibly containing some uninitialized values in the middle!
    v: SmallVec<A>,
    /// the end of the target area
    t1: usize,
    /// the start of the source area
    s0: usize,
}

impl<T: Debug, A: Array<Item = T>> Debug for InPlaceSmallVecBuilder<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let InPlaceSmallVecBuilder { s0, t1, v } = self;
        let s1 = v.len();
        let cap = v.capacity();
        write!(
            f,
            "InPlaceSmallVecBuilder(0..{},{}..{},{})",
            t1, s0, s1, cap
        )
    }
}

/// initializes the source part of this flip buffer with the given vector.
/// The target part is initially empty.
impl<A: Array> From<SmallVec<A>> for InPlaceSmallVecBuilder<A> {
    fn from(value: SmallVec<A>) -> Self {
        InPlaceSmallVecBuilder {
            v: value,
            s0: 0,
            t1: 0,
        }
    }
}

impl<A: Array> InPlaceSmallVecBuilder<A> {
    fn assert_invariants(&self) {
        assert!(self.t1 <= self.s0);
        assert!(self.s0 <= self.v.len());
        // assert!(self.v.len() <= self.v.capacity()); vec invariant
    }
    /// The current target part as a slice
    pub fn target_slice(&self) -> &[A::Item] {
        &self.v[..self.t1]
    }

    /// The current source part as a slice
    pub fn source_slice(&self) -> &[A::Item] {
        &self.v[self.s0..]
    }

    /// ensure that we have at least `capacity` space.
    fn reserve(&mut self, capacity: usize) {
        // ensure we have space!
        if self.t1 + capacity > self.s0 {
            let v = &mut self.v;
            let s0 = self.s0;
            let s1 = v.len();
            let sn = s1 - s0;
            // delegate to the underlying vec for the grow logic
            v.reserve(capacity);
            // move the source to the end of the vec
            let cap = v.capacity();
            // just move source to the end without any concern about dropping
            unsafe {
                std::ptr::copy(v.as_ptr().add(s0), v.as_mut_ptr().add(cap - sn), sn);
                v.set_len(cap);
            }
            // move s0
            self.s0 = cap - sn;
        }
    }

    /// Take at most `n` elements from `iter` to the target
    pub fn extend_from_iter<I: Iterator<Item = A::Item>>(&mut self, iter: &mut I, n: usize) {
        if n > 0 {
            self.reserve(n);
            for _ in 0..n {
                if let Some(value) = iter.next() {
                    self.push_unsafe(value)
                }
            }
        }
    }

    /// Push a single value to the target
    pub fn push(&mut self, value: A::Item) {
        // ensure we have space!
        self.reserve(1);
        self.push_unsafe(value);
    }

    fn push_unsafe(&mut self, value: A::Item) {
        unsafe { std::ptr::write(self.v.as_mut_ptr().add(self.t1), value) }
        self.t1 += 1;
    }

    /// Consume `n` elements from the source. If `take` is true they will be added to the target,
    /// else they will be dropped.
    pub fn consume(&mut self, n: usize, take: bool) {
        let n = std::cmp::min(n, self.source_slice().len());
        let v = self.v.as_mut_ptr();
        if take {
            if self.t1 != self.s0 {
                unsafe {
                    std::ptr::copy(v.add(self.s0), v.add(self.t1), n);
                }
            }
            self.t1 += n;
            self.s0 += n;
        } else {
            for _ in 0..n {
                unsafe {
                    self.s0 += 1;
                    std::ptr::drop_in_place(v.add(self.s0 - 1));
                }
            }
        }
    }

    /// Skip up to `n` elements from source without adding them to the target.
    /// They will be immediately dropped!
    pub fn skip(&mut self, n: usize) {
        let n = std::cmp::min(n, self.source_slice().len());
        let v = self.v.as_mut_ptr();
        for _ in 0..n {
            unsafe {
                self.s0 += 1;
                std::ptr::drop_in_place(v.add(self.s0 - 1));
            }
        }
    }

    /// Take up to `n` elements from source to target.
    /// If n is larger than the size of the remaining source, this will only copy all remaining elements in source.
    pub fn take(&mut self, n: usize) {
        let n = std::cmp::min(n, self.source_slice().len());
        if self.t1 != self.s0 {
            unsafe {
                let v = self.v.as_mut_ptr();
                std::ptr::copy(v.add(self.s0), v.add(self.t1), n);
            }
        }
        self.t1 += n;
        self.s0 += n;
    }

    /// Takes the next element from the source, if it exists
    pub fn pop_front(&mut self) -> Option<A::Item> {
        if self.s0 < self.v.len() {
            self.s0 += 1;
            Some(unsafe { std::ptr::read(self.v.as_ptr().add(self.s0 - 1)) })
        } else {
            None
        }
    }

    fn drop_source(&mut self) {
        // use truncate to get rid of the source part, if any, calling drop as needed
        self.v.truncate(self.s0);
        // use set_len to get rid of the gap part between t1 and s0, not calling drop!
        unsafe {
            self.v.set_len(self.t1);
        }
        self.s0 = self.t1;
        // shorten the source part
    }

    /// takes the target part of the flip buffer as a vec and drops the remaining source part, if any
    pub fn into_vec(mut self) -> SmallVec<A> {
        // drop the source part
        self.drop_source();
        // tear out the v
        let v = std::mem::replace(&mut self.v, unsafe { std::mem::uninitialized() });
        // forget the rest to prevent drop to run on uninitialized data
        std::mem::forget(self);
        v
    }
}

impl<A: Array> Drop for InPlaceSmallVecBuilder<A> {
    fn drop(&mut self) {
        // drop the source part.
        self.drop_source();
        // explicitly drop the target part.
        self.v.clear();
    }
}

#[cfg(test)]
mod tests {
    extern crate testdrop;
    use super::*;
    use testdrop::{Item, TestDrop};

    type Array<'a> = [Item<'a>; 2];

    fn everything_dropped<'a, F>(td: &'a TestDrop, n: usize, f: F)
    where
        F: Fn(SmallVec<Array<'a>>, SmallVec<Array<'a>>) -> InPlaceSmallVecBuilder<Array<'a>>,
    {
        let mut a: SmallVec<Array<'a>> = SmallVec::new();
        let mut b: SmallVec<Array<'a>> = SmallVec::new();
        let mut ids: Vec<usize> = Vec::new();
        for _ in 0..n {
            let (id, item) = td.new_item();
            a.push(item);
            ids.push(id);
        }
        for _ in 0..n {
            let (id, item) = td.new_item();
            b.push(item);
            ids.push(id);
        }
        let fb = f(a, b);
        std::mem::drop(fb);
        for id in ids {
            td.assert_drop(id);
        }
    }

    #[test]
    fn drop_just_source() {
        everything_dropped(&TestDrop::new(), 10, |a, _| a.into())
    }

    #[test]
    fn target_push_gap() {
        everything_dropped(&TestDrop::new(), 10, |a, b| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            for x in b.into_iter() {
                res.push(x);
            }
            res
        })
    }

    #[test]
    fn source_move_some() {
        everything_dropped(&TestDrop::new(), 10, |a, _| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            res.take(3);
            res
        })
    }

    #[test]
    fn source_move_all() {
        everything_dropped(&TestDrop::new(), 10, |a, _| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            res.take(10);
            res
        })
    }

    #[test]
    fn source_drop_some() {
        everything_dropped(&TestDrop::new(), 10, |a, _| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            res.skip(3);
            res
        })
    }

    #[test]
    fn source_drop_all() {
        everything_dropped(&TestDrop::new(), 10, |a, _| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            res.skip(10);
            res
        })
    }

    #[test]
    fn source_pop_some() {
        everything_dropped(&TestDrop::new(), 10, |a, _| {
            let mut res: InPlaceSmallVecBuilder<Array> = a.into();
            res.pop_front();
            res.pop_front();
            res.pop_front();
            res
        })
    }
}

/// workaround until https://github.com/servo/rust-smallvec/issues/181 is implemented
pub struct SmallVecIntoIter<A: Array> {
    data: SmallVec<A>,
    current: usize,
    end: usize,
}

impl<A: Array> Drop for SmallVecIntoIter<A> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<A: Array> Iterator for SmallVecIntoIter<A> {
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                let current = self.current as isize;
                self.current += 1;
                Some(core::ptr::read(self.data.as_ptr().offset(current)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.current;
        (size, Some(size))
    }
}

impl<A: Array> SmallVecIntoIter<A> {
    /// create a new iterator from a SmallVec
    pub fn new(data: SmallVec<A>) -> Self {
        Self {
            current: 0,
            end: data.len(),
            data,
        }
    }

    /// Returns the remaining items of this iterator as a slice.
    pub fn as_slice(&self) -> &[A::Item] {
        let len = self.end - self.current;
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().add(self.current), len) }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [A::Item] {
        let len = self.end - self.current;
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().add(self.current), len) }
    }
}
