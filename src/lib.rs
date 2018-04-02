pub trait Consumer<T> {
  type K: Copy;
  type KSet;
  fn put(&mut self, k: Self::K, t: T);
  fn advance(&mut self, ks: Self::KSet);
}
pub trait Producer<T> {
  fn get(&mut self) -> T;
}
pub trait Processor<I, O> : Consumer<I> + Producer<Option<O>> {}
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
  type K = ();
  type KSet = ();
  fn put(&mut self, _k: (), t: T) {
    mem::replace(&mut self.elem, Some(t));
  }
  fn advance(&mut self, _ks: ()) {}
}
impl<T> Producer<Option<T>> for FusedBuffer<T> {
  fn get(&mut self) -> Option<T> {
    mem::replace(&mut self.elem, None)
  }
}
impl<T> Processor<T, T> for FusedBuffer<T> {}
impl<T> Buffer<T> for FusedBuffer<T> {}

#[test]
fn fused_buffer_inserts() {
    let mut b = FusedBuffer::empty();
    assert_eq!(None, b.get());
    let i = 5;
    b.put(i);
    assert_eq!(Some(i), b.get());
    assert_eq!(None, b.get());
}

 use std::marker::PhantomData;

pub struct PreFnProcessor<I, O, F, B>
    where F: Fn(B::K, I) -> O, B: Buffer<O> {
  f: F,
  buf: B,
  _in: PhantomData<I>,
  _out: PhantomData<O>,
}
impl<I, O, F, B> PreFnProcessor<I, O, F, B>
    where F: Fn(B::K, I) -> O, B: Buffer<O> {
  pub fn new(f: F, b: B) -> PreFnProcessor<I, O, F, B> {
      PreFnProcessor { _in: PhantomData, _out: PhantomData, f: f, buf: b }
  }
}
impl<I, O, F, B> Consumer<I>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::K, I) -> O, B: Buffer<O> {
  type K = B::K;
  type KSet = B::KSet;
  fn put(&mut self, k: B::K, input: I) {
    self.buf.put(k, (self.f)(k, input));
  }
  fn advance(&mut self, _ks: B::KSet) {}
}
impl<I, O, F, B> Producer<Option<O>>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::K, I) -> O, B: Buffer<O> {
  fn get(&mut self) -> Option<O> {
    self.buf.get()
  }
}
impl<I, O, F, B> Processor<I, O>
    for PreFnProcessor<I, O, F, B>
    where F: Fn(B::K, I) -> O, B: Buffer<O> {}

#[test]
fn prefn_processor_works() {
    let mut fn_b = PreFnProcessor::new(|_k, i| i * 2, FusedBuffer::empty());
    assert_eq!(None, fn_b.get());
    let i = 5;
    fn_b.put((), i);
    assert_eq!(Some(i * 2), fn_b.get());
    assert_eq!(None, fn_b.get());
}

use std::collections::VecDeque;

pub struct LinearBuf<T>(VecDeque<T>);


