use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use abi_stable::std_types::RVec;
use core_types::Payload;
use parser::ast::{FlowStep, Pipeline, PipelineParam, Statement, Value};

use crate::outputs::PipelineOutputs;
use crate::registry::PluginRegistry;
use crate::scheduler::AutoParallelScheduler;

#[derive(Debug)]
pub enum MorflowError {
    Io(std::io::Error),
    Parse(String),
    Compile(String),
    Plugin(String),
    Execution(String),
    TypeMismatch(String),
}

impl fmt::Display for MorflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MorflowError::Io(e) => write!(f, "I/O error: {}", e),
            MorflowError::Parse(e) => write!(f, "Parse error: {}", e),
            MorflowError::Compile(e) => write!(f, "Compile error: {}", e),
            MorflowError::Plugin(e) => write!(f, "Plugin error: {}", e),
            MorflowError::Execution(e) => write!(f, "Execution error: {}", e),
            MorflowError::TypeMismatch(e) => write!(f, "Type error: {}", e),
        }
    }
}

impl std::error::Error for MorflowError {}

impl From<std::io::Error> for MorflowError {
    fn from(e: std::io::Error) -> Self {
        MorflowError::Io(e)
    }
}

/// The main entrypoint for loading and running `.morf` pipelines.
pub struct Morflow;

impl Morflow {
    /// Loads and compiles a `.morf` pipeline file from disk.
    ///
    /// Plugin search paths are automatically discovered from the environment variable
    /// `MORFLOW_ACTIONS_PATH`, binary directory, and target folders.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<MorflowPipeline, MorflowError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_str(&content)
    }

    /// Parses and compiles a `.morf` pipeline string.
    pub fn from_str(source: &str) -> Result<MorflowPipeline, MorflowError> {
        let ast = parser::parse(source).map_err(|errs| {
            MorflowError::Parse(format!(
                "Syntax error in pipeline: {}",
                errs.into_iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        // Perform strict static compile-time validation (SSA / emit rules)
        crate::validator::validate_pipeline(&ast)?;

        let registry = Arc::new(PluginRegistry::default());
        Ok(MorflowPipeline { ast, registry })
    }
}

/// An executable Morflow pipeline instance with auto-parallelization scheduling.
pub struct MorflowPipeline {
    pub ast: Pipeline,
    pub registry: Arc<PluginRegistry>,
}

impl MorflowPipeline {
    /// Returns the pipeline name if defined in the `.morf` file.
    pub fn name(&self) -> Option<&str> {
        self.ast.name.as_deref()
    }

    /// Returns the pipeline parameter declarations (`accept $param`).
    pub fn params(&self) -> &[PipelineParam] {
        &self.ast.params
    }

    /// Preloads all plugins declared across all steps in this pipeline into memory.
    pub fn warmup(&self) -> Result<(), MorflowError> {
        let action_names = collect_action_names(&self.ast.statements);
        for action in action_names {
            self.registry
                .get_or_load(&action)
                .map_err(MorflowError::Plugin)?;
        }
        Ok(())
    }

    /// Executes the pipeline with a single input payload using auto-parallelization.
    /// Returns `PipelineOutputs` containing all emitted return values.
    pub fn run(&mut self, input: Payload) -> Result<PipelineOutputs, MorflowError> {
        self.run_args(vec![input])
    }

    /// Executes the pipeline with multiple positional arguments using auto-parallelization.
    /// Returns `PipelineOutputs` containing all emitted return values.
    pub fn run_args(&mut self, args: Vec<Payload>) -> Result<PipelineOutputs, MorflowError> {
        let mut env: HashMap<String, Payload> = HashMap::new();

        // Bind input arguments to pipeline parameters
        for (i, param) in self.ast.params.iter().enumerate() {
            if let Some(arg) = args.get(i) {
                env.insert(param.name.clone(), arg.clone());
            } else if let Some(default_val) = &param.default_value {
                match default_val {
                    Value::Int(v) => {
                        env.insert(
                            param.name.clone(),
                            Payload::Data {
                                buffer: RVec::from(v.to_string().into_bytes()),
                            },
                        );
                    }
                    Value::Float(v) => {
                        env.insert(
                            param.name.clone(),
                            Payload::Data {
                                buffer: RVec::from(v.to_string().into_bytes()),
                            },
                        );
                    }
                    Value::String(v) => {
                        env.insert(
                            param.name.clone(),
                            Payload::Data {
                                buffer: RVec::from(v.as_bytes().to_vec()),
                            },
                        );
                    }
                    _ => {}
                }
            }
        }

        // If pipeline has no declared parameters but an argument was passed, bind it to $input
        if self.ast.params.is_empty() && !args.is_empty() {
            env.insert("input".to_string(), args[0].clone());
        }

        // Execute via auto-parallel scheduler
        let scheduler = AutoParallelScheduler::new(Arc::clone(&self.registry));
        scheduler.execute(&self.ast.statements, env)
    }
}

fn collect_action_names(statements: &[Statement]) -> Vec<String> {
    let mut names = Vec::new();
    for stmt in statements {
        let Statement::Flow(chain) = stmt;
        for step in &chain.steps {
            if let FlowStep::Action(call) = step {
                if call.name != "emit" && call.name != "resurface" && !names.contains(&call.name) {
                    names.push(call.name.clone());
                }
            }
        }
    }
    names
}
