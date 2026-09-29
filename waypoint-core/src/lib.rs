//! Waypoint backend: everything except the neural networks lives here.
//!
//! * [`pipeline`] orchestrates a run and emits the NDJSON event contract.
//! * [`sources`] are the imagery clients (Mapillary, Google Street View, Panoramax).
//! * [`geo`], [`sun`], [`astral`], [`ransac`] are the deterministic algorithms.
//! * [`models`] is the interface to the neural pieces; [`pyserver`] implements it
//!   over the Python model server.

pub mod astral;
pub mod geo;
pub mod models;
pub mod net;
pub mod pipeline;
pub mod pyserver;
pub mod ransac;
pub mod retrieval;
pub mod sources;
pub mod sun;
pub mod util;

pub use pipeline::{run, Mode, RunArgs, Sink};
pub use util::{Cancel, Error, Result};
