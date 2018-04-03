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
  fn get(&mut self) -> (Self::OutK, T);
}
pub trait Processor<I: Merge, O> : Consumer<I> + Producer<Option<O>> {}
pub trait Buffer<T: Merge> : Processor<T, T> {}

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
impl<T> Producer<Option<T>> for FusedBuffer<T> {
  type OutK = ();
  fn get(&mut self) -> ((), Option<T>) {
    let ret = if self.done {
        mem::replace(&mut self.elem, None)
    } else { None };
    ((), ret)
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
    assert_eq!(((), None), b.get());
    let i = 5;
    b.put(i);
    assert_eq!(((), None), b.get());
    b.advance(());
    assert_eq!(((), Some(i)), b.get());
    assert_eq!(((), None), b.get());
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
impl<I, O: Merge, F, B> Producer<Option<O>>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  type OutK = B::OutK;
  fn get(&mut self) -> (B::OutK, Option<O>) {
    self.buf.get()
  }
}
impl<I: Merge, O: Merge, F, B> Processor<I, O>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {}

#[test]
fn prefn_processor_works() {
    let mut fn_b = PreFnProcessor::new(|_k, i| i * 2, FusedBuffer::empty());
    assert_eq!(((), None), fn_b.get());
    let i = 5;
    fn_b.put((), i);
    assert_eq!(((), None), fn_b.get());
    fn_b.advance(());
    assert_eq!(((), Some(i * 2)), fn_b.get());
    assert_eq!(((), None), fn_b.get());
}

use std::collections::VecDeque;

pub struct LinearBuf<T> {
  deque: VecDeque<FusedBuffer<T>>,
  current_min: usize,
}
impl<T> LinearBuf<T> {
  pub fn new() -> LinearBuf<T> {
    LinearBuf { deque: VecDeque::new(), current_min: 0 }
  }
}
impl<T: Merge> Consumer<T> for LinearBuf<T> {
  type InK = usize;
  type KSet = (usize, usize); // Half-open, lower-inclusive interval.
  fn put(&mut self, k: usize, t: T) {
    let idx = k - self.current_min;
    let len = self.deque.len();
    if self.deque.len() > idx {
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
  fn advance(&mut self, _ks: (usize, usize)) {}
}
impl<T> Producer<Option<T>> for LinearBuf<T> {
  type OutK = ();
  fn get(&mut self) -> ((), Option<T>) {
    ((), None)
  }
}
impl<T: Merge> Processor<T, T> for LinearBuf<T> {}
impl<T: Merge> Buffer<T> for LinearBuf<T> {}


