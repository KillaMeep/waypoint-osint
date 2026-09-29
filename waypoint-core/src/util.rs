//! Small shared helpers: errors, cancellation, Python-compatible rounding,
//! bounded parallel maps that report completions as they happen.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub enum Error {
    Cancelled,
    Msg(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cancelled => write!(f, "cancelled"),
            Error::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Msg(e.to_string())
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Shared cancellation flag, checked by every loop that can take a while.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
    /// Sleep in small slices so a cancel is noticed promptly.
    pub fn sleep(&self, d: Duration) -> Result<()> {
        let end = Instant::now() + d;
        while Instant::now() < end {
            self.check()?;
            std::thread::sleep(Duration::from_millis(50).min(end - Instant::now()));
        }
        self.check()
    }
}

/// Seconds since the epoch, like Python's `time.time()`.
pub fn now_ts() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// Python's `round(x, ndigits)` for floats: correctly rounded decimal with
/// round-half-even applied to the exact binary value.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

/// Run `f` over `items` on up to `workers` threads and hand each result to
/// `on_done` on the calling thread, in completion order (the moral equivalent
/// of `ThreadPoolExecutor` + `as_completed`). `on_done` gets (completed_count,
/// item_index, result). Stops handing out new work once `cancel` fires.
pub fn par_for_each_completed<T, R, F, D>(
    items: Vec<T>,
    workers: usize,
    cancel: &Cancel,
    f: F,
    mut on_done: D,
) -> Result<()>
where
    T: Send,
    R: Send,
    F: Fn(T) -> R + Sync,
    D: FnMut(usize, usize, R),
{
    let total = items.len();
    if total == 0 {
        return Ok(());
    }
    let queue: Mutex<VecDeque<(usize, T)>> = Mutex::new(items.into_iter().enumerate().collect());
    let (tx, rx) = mpsc::channel::<(usize, R)>();
    let n_threads = workers.max(1).min(total);
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let tx = tx.clone();
            let queue = &queue;
            let f = &f;
            s.spawn(move || loop {
                if cancel.is_cancelled() {
                    break;
                }
                let job = queue.lock().unwrap().pop_front();
                let Some((i, item)) = job else { break };
                let r = f(item);
                if tx.send((i, r)).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        let mut completed = 0usize;
        for (i, r) in rx {
            completed += 1;
            on_done(completed, i, r);
        }
    });
    cancel.check()
}

/// Tiny xorshift RNG for the few places that need a random subset (not
/// cryptographic, not reproducible against numpy by design).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    pub fn from_time() -> Self {
        Rng::new(SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Choose `k` distinct indices out of `n` (partial Fisher-Yates), unordered like `np.random.choice`.
    pub fn choice_no_replace(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k.min(n) {
            let j = i + self.below(n - i);
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_rounding() {
        assert_eq!(py_round(0.0625, 3), 0.062); // exact tie -> even
        assert_eq!(py_round(2.675, 2), 2.67); // binary repr is below the tie
        assert_eq!(py_round(0.9375, 3), 0.938);
        assert_eq!(py_round(41.12215042114258, 3), 41.122);
    }

    #[test]
    fn par_completes_everything() {
        let c = Cancel::new();
        let mut seen = vec![];
        par_for_each_completed((0..100).collect::<Vec<_>>(), 8, &c, |x| x * 2, |_n, i, r| seen.push((i, r))).unwrap();
        seen.sort();
        assert_eq!(seen.len(), 100);
        assert!(seen.iter().all(|(i, r)| *r == *i * 2));
    }
}
