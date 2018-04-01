pub trait Consumer<T> {
  fn put(&mut self, t: T);
}
pub trait Producer<T> {
  fn get(&mut self) -> T;
}
pub trait Processor<I, O> : Consumer<I> + Producer<Option<O>> {}
pub type Buffer<T> = Processor<T, T>;

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
}
impl<T> Consumer<T> for FusedBuffer<T> {
  fn put(&mut self, t: T) {
    mem::replace(&mut self.elem, Some(t));
  }
}
impl<T> Producer<Option<T>> for FusedBuffer<T> {
  fn get(&mut self) -> Option<T> {
    mem::replace(&mut self.elem, None)
  }
}
 use std::marker::PhantomData;

pub struct PreFnProcessor<I, O, F: Fn(I) -> O> {
  _in: PhantomData<I>,
  f: F,
  buf: FusedBuffer<O>
}
impl<I, O, F: Fn(I) -> O> PreFnProcessor<I, O, F> {
  pub fn new(f_: F, b: FusedBuffer<O>) -> PreFnProcessor<I, O, F> {
      PreFnProcessor { _in: PhantomData, f: f_, buf: b }
  }
}
impl<I, O, F: Fn(I) -> O> Consumer<I>
    for PreFnProcessor<I, O, F> {
  fn put(&mut self, input: I) {
    mem::replace(&mut self.buf.elem, Some((self.f)(input)));
  }
}
impl<I, O, F: Fn(I) -> O> Producer<Option<O>>
    for PreFnProcessor<I, O, F> {
  fn get(&mut self) -> Option<O> {
    mem::replace(&mut self.buf.elem, None)
  }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_buffer_inserts() {
        let mut b = FusedBuffer::empty();
        assert_eq!(None, b.get());
        let i = 5;
        b.put(i);
        assert_eq!(Some(i), b.get());
        assert_eq!(None, b.get());
    }

    #[test]
    fn prefn_processor_works() {
        let mut fn_b = PreFnProcessor::new(|i| i * 2, FusedBuffer::empty());
        assert_eq!(None, fn_b.get());
        let i = 5;
        fn_b.put(i);
        assert_eq!(Some(i * 2), fn_b.get());
        assert_eq!(None, fn_b.get());

    }
}
