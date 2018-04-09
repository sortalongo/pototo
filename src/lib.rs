pub trait Merge {
  fn merge(self, other: Self) -> Self;
}
pub trait Consumer<T: Merge> {
  type InK: Copy;
  type KSet;
  fn put(&mut self, k: Self::InK, t: T);
  fn advance(&mut self, ks: Self::KSet);
}
pub trait Producer<T> {
  type OutK: Copy;
  fn get(&mut self) -> Option<(Self::OutK, T)>;
}
pub trait Processor<I: Merge, O> : Consumer<I> + Producer<O> {}
pub trait Buffer<T: Merge> : Processor<T, T> {}

pub trait PointSet {
  type KSet;
  fn empty() -> Self::KSet;
  fn merge_sets(k1: Self::KSet, k2: Self::KSet) -> Self::KSet;
}

use std::cmp::Ord;
use std::fmt::Debug;
#[derive(Copy, Clone, Debug, Eq, Ord, PartialOrd, PartialEq)]
struct LowerNat(usize);

impl PointSet for LowerNat {
  type KSet = LowerNat;
  fn empty() -> LowerNat { LowerNat(0) }
  fn merge_sets(n1: LowerNat, n2: LowerNat) -> LowerNat {
    std::cmp::max(n1, n2)
  }
}

use std::mem;

pub struct FusedBuffer<T> {
  elem: Option<T>,
  done: bool,
}
impl<T> FusedBuffer<T> {
  pub fn new(t: T) -> FusedBuffer<T> {
      FusedBuffer { elem: Some(t), done: false }
  }
  pub fn empty() -> FusedBuffer<T> {
      FusedBuffer { elem: None, done: false }
  }
  pub fn put(&mut self, t: T) where T: Merge {
      Consumer::put(self, (), t);
  }
}
impl<T: Merge> Consumer<T> for FusedBuffer<T> {
  type InK = ();
  type KSet = ();
  fn put(&mut self, _k: (), t: T) {
    if self.done {
        panic!("Trying to put into a finished FusedBuffer.");
    }
    let tmp =
      if self.elem.is_some() {
        let mut prev = None;
        mem::swap(&mut prev, &mut self.elem);
        prev.unwrap().merge(t)
      } else {
        t
      };
    mem::replace(&mut self.elem, Some(tmp));
  }
  fn advance(&mut self, _ks: ()) {
    self.done = true;
  }
}
impl<T> Producer<T> for FusedBuffer<T> {
  type OutK = ();
  fn get(&mut self) -> Option<((), T)> {
    if self.done {
        mem::replace(&mut self.elem, None).map(|elem| ((), elem))
    } else { None }
  }
}
impl<T: Merge> Processor<T, T> for FusedBuffer<T> {}
impl<T: Merge> Buffer<T> for FusedBuffer<T> {}

impl Merge for i64 {
  fn merge(self, other: i64) -> i64 {
    self + other
  }
}
#[test]
fn fused_buffer_inserts() {
    let mut b = FusedBuffer::empty();
    assert_eq!(None, b.get());
    let i = 5;
    b.put(i);
    assert_eq!(None, b.get());
    b.advance(());
    assert_eq!(Some(((), i)), b.get());
    assert_eq!(None, b.get());
}

 use std::marker::PhantomData;

pub struct PreFnProcessor<I, O, F, B>
    where O: Merge, F: Fn(B::InK, I) -> O, B: Buffer<O> {
  f: F,
  buf: B,
  // Below are just markers for the compiler to track unused types.
  _in: PhantomData<I>,
  _out: PhantomData<O>,
}
impl<I, O: Merge, F, B> PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  pub fn new(f: F, b: B) -> PreFnProcessor<I, O, F, B> {
      PreFnProcessor { _in: PhantomData, _out: PhantomData, f: f, buf: b }
  }
}
impl<I: Merge, O: Merge, F, B> Consumer<I>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  type InK = B::InK;
  type KSet = B::KSet;
  fn put(&mut self, k: B::InK, input: I) {
    self.buf.put(k, (self.f)(k, input));
  }
  fn advance(&mut self, ks: B::KSet) {
    self.buf.advance(ks);
  }
}
impl<I, O: Merge, F, B> Producer<O>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  type OutK = B::OutK;
  fn get(&mut self) -> Option<(B::OutK, O)> {
    self.buf.get()
  }
}
impl<I: Merge, O: Merge, F, B> Processor<I, O>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {}

#[test]
fn prefn_processor_works() {
    let mut fn_b = PreFnProcessor::new(|_k, i| i * 2, FusedBuffer::empty());
    assert_eq!(None, fn_b.get());
    let i = 5;
    fn_b.put((), i);
    assert_eq!(None, fn_b.get());
    fn_b.advance(());
    assert_eq!(Some(((), i * 2)), fn_b.get());
    assert_eq!(None, fn_b.get());
}

use std::collections::VecDeque;

pub struct LinearBuf<T> {
  deque: VecDeque<FusedBuffer<T>>,
  buffer_min: usize,
  complete_min: usize,
}
impl<T> LinearBuf<T> {
  pub fn new() -> LinearBuf<T> {
    LinearBuf { deque: VecDeque::new(), buffer_min: 0, complete_min: 0 }
  }
}

impl<T: Merge + Debug> Consumer<T> for LinearBuf<T> {
  type InK = usize;
  type KSet = usize; // minimum unfinished index.
  fn put(&mut self, k: usize, t: T) {
    assert!(k >= self.buffer_min);
    assert!(k >= self.complete_min);
    let idx = k - self.buffer_min;
    let len = self.deque.len();
    if len > idx {
      if let Some(buf) = self.deque.get_mut(idx) {
        buf.put(t);
      } else {
          panic!("Failed getting index {idx} from {self.deque}");
      }
    } else {
      self.deque.reserve(idx - len + 1);
      for _j in len..idx {
        self.deque.push_back(FusedBuffer::empty());
      }
      self.deque.push_back(FusedBuffer::new(t));
    }
  }
  fn advance(&mut self, ks: usize) {
    self.complete_min = ks;
  }
}
impl<T> Producer<T> for LinearBuf<T> {
  type OutK = usize;
  fn get(&mut self) -> Option<(usize, T)> {
    let pending = self.complete_min as i64 - self.buffer_min as i64;
    assert!(pending >= 0);
    if pending == 0 {
        return None;
    }
    let to_skip = self.deque.iter().take(pending as usize).take_while(|fb| {
        fb.elem.is_none()
    }).count();
    let buffer_min = self.buffer_min;
    self.buffer_min += to_skip + 1;
    self.deque.drain(0..(to_skip + 1)).last()
        .and_then(|fb| fb.elem)
        .map(|elem| (buffer_min + to_skip, elem))
  }
}
impl<T: Merge + Debug> Processor<T, T> for LinearBuf<T> {}
impl<T: Merge + Debug> Buffer<T> for LinearBuf<T> {}

impl Merge for String {
  fn merge(self, other: String) -> String {
    self + &other
  }
}

#[test]
fn linear_buf_works() {
  let mut buf = LinearBuf::<String>::new();
  buf.put(1, "1".to_string());
  buf.put(2, "2".to_string());
  buf.put(0, "0".to_string());
  assert_eq!(None, buf.get());
  buf.advance(1);
  assert_eq!(Some((0, "0".to_string())), buf.get());
  assert_eq!(None, buf.get());
  buf.advance(3);
  assert_eq!(Some((1, "1".to_string())), buf.get());
  assert_eq!(Some((2, "2".to_string())), buf.get());
  assert_eq!(None, buf.get());
}

use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Copy, Clone, Debug, Eq, Ord, PartialOrd, PartialEq)]
pub enum Boundary<K> { Open(K), Closed(K) }
impl<K> Boundary<K> {
  pub fn value(&self) -> &K {
    match self {
      &Boundary::Open(ref k) => k,
      &Boundary::Closed(ref k) => k,
    }
  }
  pub fn is_open(&self) -> bool {
    match self {
      &Boundary::Open(ref _k) => true,
      &Boundary::Closed(ref _k) => false,
    }
  }
  pub fn inverted(self) -> Self {
    match self {
      Boundary::Open(k) => Boundary::Closed(k),
      Boundary::Closed(k) => Boundary::Open(k),
    }
  }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialOrd, PartialEq)]
pub struct Interval<K: Ord> {
  lower: Boundary<K>,
  upper: Boundary<K>
}
impl<K> Interval<K> where K: Ord + Clone + Debug {
  fn check_order(lower: &Boundary<K>, upper: &Boundary<K>) {
    if lower.value() > upper.value() {
      panic!("Interval: {lower} must be <= {upper}.");
    } else if lower.value() == upper.value()
        && lower.is_open() != upper.is_open() {
      panic!("Interval: Constructing illegal half-open interval with equal endpoints {lower.value()}.");
    }
  }

  pub fn empty() -> Interval<K> where K: Default {
    Interval::open(Default::default(), Default::default())
  }
  pub fn singleton(k: K) -> Interval<K> {
    Interval::closed(k.clone(), k)
  }

  pub fn of(lower: Boundary<K>, upper: Boundary<K>) -> Interval<K> {
    Interval::check_order(&lower, &upper);
    Interval { lower: lower, upper: upper, }
  }
  pub fn inverted(lower: Boundary<K>, upper: Boundary<K>) -> Interval<K>  {
    Interval::of(lower.inverted(), upper.inverted())
  }

  pub fn open(lower: K, upper: K) -> Interval<K> {
    Interval::of(Boundary::Open(lower), Boundary::Open(upper))
  }
  pub fn closed(lower: K, upper: K) -> Interval<K> {
    Interval::of(Boundary::Closed(lower), Boundary::Closed(upper))
  }
  pub fn closed_open(lower: K, upper: K) -> Interval<K> {
    Interval::of(Boundary::Closed(lower), Boundary::Open(upper))
  }
  pub fn open_closed(lower: K, upper: K) -> Interval<K> {
    Interval::of(Boundary::Open(lower), Boundary::Closed(upper))
  }

  pub fn is_empty(&self) -> bool {
    self.upper.value() == self.lower.value()
        && self.upper.is_open() && self.lower.is_open()
  }

  pub fn overlaps_with(&self, other: &Interval<K>) -> bool {
      if self.is_empty() || other.is_empty() {
        return false;
      }
      let is_self_lower = self.lower.value() < other.lower.value();
      let (smaller, larger) = if is_self_lower {
        (self, other)
      } else {
        (other, self)
      };
      let is_range_overlap = smaller.upper.value() > larger.lower.value();
      let is_point_overlap =
          !Interval::inverted(smaller.upper.clone(), larger.lower.clone())
            .is_empty();
      is_range_overlap || is_point_overlap
  }

  pub fn merge_overlapping(self, other: Interval<K>) -> Interval<K> {
      assert!(self.overlaps_with(&other));
      let is_self_lower = self.lower.value() < other.lower.value()
          || (self.lower.value() == other.lower.value()
              && !self.lower.is_open());
      let is_self_upper = self.upper.value() > other.upper.value()
          || (self.upper.value() == other.upper.value()
              && !self.upper.is_open());
      match (is_self_lower, is_self_upper) {
        (true, true) => self,
        (true, false) => Interval::of(self.lower, other.upper),
        (false, false) => other,
        (false, true) => Interval::of(other.lower, self.upper),
      }
  }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntervalSet<K> where K: Clone + Ord {
  // IntervalSet doesn't allow overlapping intervals, so we don't need an
  // interval tree for efficiency.
  intervals: BTreeSet<Interval<K>>
}
impl<K> IntervalSet<K> where K: Clone + Debug + Ord {
  pub fn empty() -> IntervalSet<K> {
    IntervalSet { intervals: BTreeSet::new() }
  }
  pub fn singleton(i: Interval<K>) -> IntervalSet<K> {
    let mut s = BTreeSet::new();
    s.insert(i);
    IntervalSet { intervals: s }
  }

  // Inserts an interval, merging with any overlapping it.
  pub fn insert(&mut self, int: Interval<K>) {
    let lower_singleton = Interval::singleton(int.lower.value().clone());
    fn filter_opt<T, F: FnOnce(&T) -> bool>(
        opt: Option<T>, f: F) -> Option<T> {
      opt.and_then(|t| if f(&t) { Some(t) } else { None })
    }
    let prev_opt =
        filter_opt(self.intervals.range(..lower_singleton).next_back(),
            |prev| int.overlaps_with(prev)).cloned();
    let upper_singleton = Interval::singleton(int.upper.value().clone());
    let next_opt =
        filter_opt(self.intervals.range(upper_singleton..).next(),
            |next| int.overlaps_with(next)).cloned();

    for prev in prev_opt.iter() {
      assert!(self.intervals.remove(prev));
    }
    for next in next_opt.iter() {
      assert!(self.intervals.remove(next));
    }
    let prev_int_merged = match prev_opt {
      Some(prev) => prev.merge_overlapping(int),
      None => int,
    };
    let all_merged = match next_opt {
        Some(next) => prev_int_merged.merge_overlapping(next),
        None => prev_int_merged,
    };
    self.intervals.insert(all_merged);
  }

  // Returns the union of two IntervalSets. The returned IntervalSet:
  // 1. Contains no overlapping intervals.
  // 2. Contains only intervals overlapping with at least one interval in the inputs.
  // 3. Contains no intervals containing regions not overlapping any intervals in the inputs.
  pub fn union(self, other: IntervalSet<K>) -> IntervalSet<K>  {
    let (smaller, mut larger) =
        if self.intervals.len() > other.intervals.len() {
          (other, self)
        } else { (self, other) };
    for int in smaller.intervals {
      larger.insert(int);
    }
    larger
  }
}

impl<K> PointSet for K where K: Ord + Clone + Debug + Default {
  type KSet = IntervalSet<K>;
  fn empty() -> IntervalSet<K> {
    IntervalSet::empty()
  }
  fn merge_sets(ints1: IntervalSet<K>, ints2: IntervalSet<K>)
      -> IntervalSet<K> {
    ints1.union(ints2)
  }
}


pub struct ParBuf<K, V>
    where K: Ord + PointSet {
  map: BTreeMap<K, V>,
  completed: K::KSet,
}

