//! Waypoint backend: the pipeline, its algorithms, and in-process inference (ONNX Runtime).
//!
//! * [`pipeline`] orchestrates a run and emits the NDJSON event contract.
//! * [`sources`] are the imagery clients (Mapillary, Google Street View, Panoramax).
//! * [`geo`], [`sun`], [`astral`], [`ransac`] are the deterministic algorithms.
//! * [`models`] is the interface to the neural pieces; [`native`] implements it
//!   on ONNX Runtime.

pub mod astral;
pub mod geo;
pub mod imgio;
pub mod models;
pub mod native;
pub mod net;
pub mod onnx;
pub mod pipeline;
pub mod plonk;
pub mod prep;
pub mod ransac;
pub mod retrieval;
pub mod sources;
pub mod sun;
pub mod util;

pub use pipeline::{run, Mode, RunArgs, Sink};
pub use util::{Cancel, Error, Result};
