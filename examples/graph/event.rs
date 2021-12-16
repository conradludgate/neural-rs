use std::io;
use std::sync::mpsc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use termion::event::Key;
use termion::input::TermRead;

use crate::train::train;

pub enum Event {
    Input(Key),
    Tick,
    EpochComplete(f64),
}

/// A small event handler that wrap termion input and tick events. Each event
/// type is handled in its own thread and returned to a common `Receiver`
pub struct Events {
    rx: mpsc::Receiver<Event>,
    _input_handle: thread::JoinHandle<()>,
    _ignore_exit_key: Arc<AtomicBool>,
    _tick_handle: thread::JoinHandle<()>,
    _train_handle: thread::JoinHandle<()>,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub exit_key: Key,
    pub tick_rate: Duration,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            exit_key: Key::Char('q'),
            tick_rate: Duration::from_millis(250),
        }
    }
}

impl Events {
    pub fn new() -> Events {
        Events::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Events {
        let (tx, rx) = mpsc::channel();
        let _ignore_exit_key = Arc::new(AtomicBool::new(false));
        let _input_handle = {
            let tx = tx.clone();
            let ignore_exit_key = _ignore_exit_key.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                for key in stdin.keys().flatten() {
                    if let Err(err) = tx.send(Event::Input(key)) {
                        eprintln!("{}", err);
                        return;
                    }
                    if !ignore_exit_key.load(Ordering::Relaxed) && key == config.exit_key {
                        return;
                    }
                }
            })
        };
        let _tick_handle = {
            let tx = tx.clone();
            thread::spawn(move || loop {
                if tx.send(Event::Tick).is_err() {
                    break;
                }
                thread::sleep(config.tick_rate);
            })
        };
        let _train_handle = {
            thread::spawn(move || {
                train(tx);
            })
        };
        Events {
            rx,
            _ignore_exit_key,
            _input_handle,
            _tick_handle,
            _train_handle,
        }
    }

    pub fn next(&self) -> Result<Event, mpsc::RecvError> {
        self.rx.recv()
    }
}
