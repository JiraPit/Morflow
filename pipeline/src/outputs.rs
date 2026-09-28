use std::collections::HashMap;
use std::ops::{Deref, Index};

use core_types::Payload;

use crate::engine::MorflowError;

/// Container for output payloads emitted from a pipeline execution back to the host.
///
/// In Morflow, flows that return values to the host terminate with the `emit` action:
/// - Single return: `... >> emit` or `... >> emit("audio")`
/// - Multiple returns: `... >> emit("left")` and `... >> emit("right")`
#[derive(Debug, Clone, Default)]
pub struct PipelineOutputs {
    outputs: HashMap<String, Payload>,
}

impl PipelineOutputs {
    /// Creates a new `PipelineOutputs` instance from a map of named payloads.
    pub fn new(outputs: HashMap<String, Payload>) -> Self {
        Self { outputs }
    }

    /// Returns the single emitted payload (if exactly one flow emitted),
    /// or returns an error if 0 or multiple flows emitted.
    pub fn into_single(self) -> Result<Payload, MorflowError> {
        if self.outputs.len() == 1 {
            let (_, payload) = self.outputs.into_iter().next().unwrap();
            Ok(payload)
        } else if self.outputs.is_empty() {
            Err(MorflowError::Execution(
                "Pipeline did not emit any outputs (ensure returning flows end with '>> emit')"
                    .to_string(),
            ))
        } else {
            let keys: Vec<String> = self.outputs.keys().cloned().collect();
            Err(MorflowError::Execution(format!(
                "Expected a single emitted output, but pipeline emitted {} outputs: [{}]. Access individual outputs by name via .get(\"name\") or [\"name\"].",
                self.outputs.len(),
                keys.join(", ")
            )))
        }
    }

    /// Returns a reference to the single emitted payload (if exactly one flow emitted).
    pub fn single(&self) -> Result<&Payload, MorflowError> {
        if self.outputs.len() == 1 {
            let (_, payload) = self.outputs.iter().next().unwrap();
            Ok(payload)
        } else if self.outputs.is_empty() {
            Err(MorflowError::Execution(
                "Pipeline did not emit any outputs (ensure returning flows end with '>> emit')"
                    .to_string(),
            ))
        } else {
            let keys: Vec<String> = self.outputs.keys().cloned().collect();
            Err(MorflowError::Execution(format!(
                "Expected a single emitted output, but pipeline emitted {} outputs: [{}]. Access individual outputs by name via .get(\"name\") or [\"name\"].",
                self.outputs.len(),
                keys.join(", ")
            )))
        }
    }

    /// Returns a reference to the payload emitted under `name`.
    pub fn get(&self, name: &str) -> Option<&Payload> {
        self.outputs.get(name)
    }

    /// Returns a mutable reference to the payload emitted under `name`.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Payload> {
        self.outputs.get_mut(name)
    }

    /// Takes and removes the payload emitted under `name`.
    pub fn take(&mut self, name: &str) -> Option<Payload> {
        self.outputs.remove(name)
    }

    /// Returns `true` if an output with the given name was emitted.
    pub fn contains_key(&self, name: &str) -> bool {
        self.outputs.contains_key(name)
    }

    /// Returns the number of emitted outputs.
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    /// Returns `true` if no outputs were emitted.
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Access the underlying `HashMap<String, Payload>`.
    pub fn as_map(&self) -> &HashMap<String, Payload> {
        &self.outputs
    }

    /// Consumes this struct and returns the underlying `HashMap<String, Payload>`.
    pub fn into_map(self) -> HashMap<String, Payload> {
        self.outputs
    }
}

impl Deref for PipelineOutputs {
    type Target = HashMap<String, Payload>;

    fn deref(&self) -> &Self::Target {
        &self.outputs
    }
}

impl Index<&str> for PipelineOutputs {
    type Output = Payload;

    fn index(&self, name: &str) -> &Self::Output {
        &self.outputs[name]
    }
}

impl IntoIterator for PipelineOutputs {
    type Item = (String, Payload);
    type IntoIter = std::collections::hash_map::IntoIter<String, Payload>;

    fn into_iter(self) -> Self::IntoIter {
        self.outputs.into_iter()
    }
}
