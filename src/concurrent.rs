use std::sync::mpsc::{channel, Sender, Receiver};
use std::boxed::FnBox;

pub struct ConcurrentCalc<T, U> {
    sender: Sender<(T, Box<FnBox<(T,), Output = U> + Send>, Sender<U>)>,
}

impl<T: Send + 'static, U: Send + 'static> ConcurrentCalc<T, U> {
    pub fn new() -> ConcurrentCalc<T, U> {
        let (send, recv) = channel();
        ::std::thread::spawn(move || {
            for message in recv.iter() {
                let (data, f, s): (T, Box<FnBox<(T,), Output = U> + Send>, Sender<_>) = message;
                s.send(f.call_box((data,))).unwrap(); // Crash if sending fails.
            }
        });
        ConcurrentCalc { sender: send }
    }
    pub fn calculate<F>(&mut self, data: T, f: F) -> Receiver<U>
        where F: FnOnce<(T,), Output = U>,
              F: Send + 'static
    {
        let (send, recv) = channel();
        self.sender.send((data, Box::new(f), send)).unwrap(); // Crash if sending fails
        recv
    }
}

#[test]
fn testConcurrent() {
    let mut cc: ConcurrentCalc<u8, u8> = ConcurrentCalc::new();
    let mut recv = cc.calculate(3, move |x| x + 1);
    assert!(recv.recv().unwrap() == 4);
    let mut recv = cc.calculate(10, move |x| x - 4);
    assert!(future.recv().unwrap() == 6);
}
