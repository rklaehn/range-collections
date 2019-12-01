use criterion::{black_box, criterion_group, criterion_main, Criterion};
use range_collections::RangeSet;

type Elem = u64;

fn union_new(a: &RangeSet<Elem>, b: &RangeSet<Elem>) -> RangeSet<Elem> {
    a | b
}

fn intersection_new(a: &RangeSet<Elem>, b: &RangeSet<Elem>) -> RangeSet<Elem> {
    a & b
}

fn intersects_new(a: &RangeSet<Elem>, b: &RangeSet<Elem>) -> bool {
    !a.is_disjoint(b)
}

fn elem(i: usize) -> Elem {
    i as u64
}

pub fn interleaved(c: &mut Criterion) {
    let mut a: RangeSet<Elem> = RangeSet::empty();
    let mut b: RangeSet<Elem> = RangeSet::empty();
    let n = 10000;
    for i in 0..n {
        let j = i * 2;
        a |= RangeSet::from(elem(j)..elem(j + 1));
        b |= RangeSet::from(elem(j + 1)..elem(j + 2));
    }
    c.bench_function("union_interleaved_new", |bencher| {
        bencher.iter(|| union_new(black_box(&a), black_box(&b)))
    });
    c.bench_function("intersection_interleaved_new", |bencher| {
        bencher.iter(|| intersection_new(black_box(&a), black_box(&b)))
    });
    c.bench_function("intersects_interleaved_new", |bencher| {
        bencher.iter(|| intersects_new(black_box(&a), black_box(&b)))
    });
}

pub fn disjoint(c: &mut Criterion) {
    let mut a: RangeSet<Elem> = RangeSet::empty();
    let mut b: RangeSet<Elem> = RangeSet::empty();
    let n = 10000;
    for i in 0..n {
        let j = i + n;
        a |= RangeSet::from(elem(i * 2)..elem(i * 2 + 1));
        b |= RangeSet::from(elem(j * 2)..elem(j * 2 + 1));
    }
    c.bench_function("union_disjoint_new", |bencher| {
        bencher.iter(|| union_new(black_box(&a), black_box(&b)))
    });
    c.bench_function("intersection_disjoint_new", |bencher| {
        bencher.iter(|| intersection_new(black_box(&a), black_box(&b)))
    });
    c.bench_function("intersects_disjoint_new", |bencher| {
        bencher.iter(|| intersects_new(black_box(&a), black_box(&b)))
    });
}

criterion_group!(benches, interleaved, disjoint);
criterion_main!(benches);
