extern crate sync;

use std::{comm};
use sync::Future;

pub struct ConcurrentCalc<T,U> {
  sender: comm::Sender<(T,proc(T):Send->U,comm::Sender<U>)>
}

impl<T:Send, U:Send> ConcurrentCalc<T,U> {
  pub fn new() -> ConcurrentCalc<T,U> {
    let (send, recv) = comm::channel();
    spawn(proc() {
      loop {
        match recv.recv_opt().ok() {
          Some(message) => {
            let (data,f,s): (T,proc(T):Send->U,comm::Sender<U>) = message;
            s.send(f(data))
          },
          None => break
        }
      }
    });
    ConcurrentCalc{ sender: send }
  }
  pub fn calculate(&mut self, data: T, f: proc(T):Send->U) -> Future<U> {
    let (send, recv) = comm::channel();
    self.sender.send( (data,f,send) );
    Future::from_fn(proc(){recv.recv()})
  }
}

#[test]
fn testConcurrent() {
  let mut cc: ConcurrentCalc<uint,uint> = ConcurrentCalc::new();
  let mut future = cc.calculate(3, proc(x) {x+1});
  assert!(future.get() == 4);
  let mut future = cc.calculate(10, proc(x) {x-4});
  assert!(future.get() == 6);
}

