use std::sync::mpsc::{channel, Sender};
use std::sync::{Future};
use std::thunk::{Thunk};

pub struct ConcurrentCalc<T,U> {
  sender: Sender<(T,Thunk<T,U>, Sender<U>)>
}

impl<T:Send, U:Send> ConcurrentCalc<T,U> {
  pub fn new() -> ConcurrentCalc<T,U> {
    let (send, recv) = channel();
    ::std::thread::Thread::spawn(move || {
      for message in recv.iter() {
        let (data,f,s): (T,Thunk<T,U>, Sender<_>) = message;
        s.send(f.invoke(data)).unwrap(); // Crash if sending fails.
      }      
    }).detach();
    ConcurrentCalc{ sender: send }
  }
  pub fn calculate<F: FnOnce(T) -> U + Send>(&mut self, data: T, f: F) -> Future<U> {
    let (send, recv) = channel();
    self.sender.send( (data, Thunk::with_arg(f), send) ).unwrap(); // Crash if sending fails
    Future::from_receiver(recv)
  }
}

#[test]
fn testConcurrent() {
  let mut cc: ConcurrentCalc<u8,u8> = ConcurrentCalc::new();
  let mut future = cc.calculate(3, move |x| {x+1});
  assert!(future.get() == 4);
  let mut future = cc.calculate(10, move |x| {x-4});
  assert!(future.get() == 6);
}

