use rand::prelude::*;
use range_collections::RangeSet;
use std::ops::{Bound, Range};

fn create_messages(n: usize, delay: usize) -> Vec<Range<usize>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut msgs: Vec<Range<usize>> = Vec::new();
    let mut offset = 0;

    // create some random sized messages
    for _ in 0..n {
        let len = rng.gen::<usize>() % 10 + 1;
        msgs.push(Range {
            start: offset,
            end: offset + len,
        });
        offset += len;
    }

    // "delay" some of them by randomly swapping with the successor
    for _ in 0..delay {
        for i in 1..msgs.len() {
            if rng.gen::<bool>() {
                msgs.swap(i - 1, i);
            }
        }
    }
    msgs
}

fn test(msgs: &Vec<Range<usize>>) -> RangeSet<usize> {
    let mut buffer: RangeSet<usize> = RangeSet::from(..0);
    for msg in msgs.iter().cloned() {
        buffer |= RangeSet::from(msg);
    }
    buffer
}

fn main() {
    let msgs = create_messages(1000000, 5);
    // do it a few times for a nice flamegraph
    for _ in 0..100 {
        test(&msgs);
    }

    let mut buffer: RangeSet<usize> = RangeSet::from(..0);
    for (i, msg) in msgs.into_iter().enumerate() {
        buffer |= RangeSet::from(msg);
        if (i % 1000) == 0 {
            if let Some((_, Bound::Excluded(end))) = buffer.iter().next() {
                println!(
                    "After {} msgs, the last contiguous sequence number is {:?}",
                    i,
                    end - 1
                );
            }
        }
    }
    println!("{:?}", buffer);
    println!("{:?}", buffer.boundaries().capacity());
}
