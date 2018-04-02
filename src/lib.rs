pub trait Consumer<T> {
  type InK: Copy;
  type KSet;
  fn put(&mut self, k: Self::InK, t: T);
  fn advance(&mut self, ks: Self::KSet);
}
pub trait Producer<T> {
  type OutK: Copy;
  fn get(&mut self) -> (Self::OutK, T);
}
pub trait Processor<I, O> : Consumer<I> + Producer<Option<O>> {

}
pub trait Buffer<T> : Processor<T, T> {}

use std::mem;

pub struct FusedBuffer<T> {
  elem: Option<T>
}
impl<T> FusedBuffer<T> {
  pub fn new(t: T) -> FusedBuffer<T> {
      FusedBuffer { elem: Some(t) }
  }
  pub fn empty() -> FusedBuffer<T> {
      FusedBuffer { elem: None }
  }
  pub fn put(&mut self, t: T) {
      Consumer::put(self, (), t);
  }
}
impl<T> Consumer<T> for FusedBuffer<T> {
  type InK = ();
  type KSet = ();
  fn put(&mut self, _k: (), t: T) {
    mem::replace(&mut self.elem, Some(t));
  }
  fn advance(&mut self, _ks: ()) {}
}
impl<T> Producer<Option<T>> for FusedBuffer<T> {
  type OutK = ();
  fn get(&mut self) -> ((), Option<T>) {
    ((), mem::replace(&mut self.elem, None))
  }
}
impl<T> Processor<T, T> for FusedBuffer<T> {}
impl<T> Buffer<T> for FusedBuffer<T> {}

#[test]
fn fused_buffer_inserts() {
    let mut b = FusedBuffer::empty();
    assert_eq!(((), None), b.get());
    let i = 5;
    b.put(i);
    assert_eq!(((), Some(i)), b.get());
    assert_eq!(((), None), b.get());
}

 use std::marker::PhantomData;

pub struct PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  f: F,
  buf: B,
  // Below are just markers for the compiler to track unused types.
  _in: PhantomData<I>,
  _out: PhantomData<O>,
}
impl<I, O, F, B> PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  pub fn new(f: F, b: B) -> PreFnProcessor<I, O, F, B> {
      PreFnProcessor { _in: PhantomData, _out: PhantomData, f: f, buf: b }
  }
}
impl<I, O, F, B> Consumer<I>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  type InK = B::InK;
  type KSet = B::KSet;
  fn put(&mut self, k: B::InK, input: I) {
    self.buf.put(k, (self.f)(k, input));
  }
  fn advance(&mut self, _ks: B::KSet) {}
}
impl<I, O, F, B> Producer<Option<O>>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {
  type OutK = B::OutK;
  fn get(&mut self) -> (B::OutK, Option<O>) {
    self.buf.get()
  }
}
impl<I, O, F, B> Processor<I, O>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::InK, I) -> O, B: Buffer<O> {}

#[test]
fn prefn_processor_works() {
    let mut fn_b = PreFnProcessor::new(|_k, i| i * 2, FusedBuffer::empty());
    assert_eq!(((), None), fn_b.get());
    let i = 5;
    fn_b.put((), i);
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
impl<T> Consumer<T> for LinearBuf<T> {
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
impl<T> Processor<T, T> for LinearBuf<T> {}
impl<T> Buffer<T> for LinearBuf<T> {}


