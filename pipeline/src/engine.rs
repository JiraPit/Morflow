use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use abi_stable::std_types::RVec;
use core_types::Payload;
use parser::ast::{FlowStep, Pipeline, PipelineParam, Statement, Value};

use crate::outputs::PipelineOutputs;
use crate::registry::ActionRegistry;
use crate::scheduler::AutoParallelScheduler;

#[derive(Debug)]
pub enum MorflowError {
    Io(std::io::Error),
    Parse(String),
    Compile(String),
    Action(String),
    Execution(String),
    TypeMismatch(String),
}

impl fmt::Display for MorflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MorflowError::Io(e) => write!(f, "I/O error: {}", e),
            MorflowError::Parse(e) => write!(f, "Parse error: {}", e),
            MorflowError::Compile(e) => write!(f, "Compile error: {}", e),
            MorflowError::Action(e) => write!(f, "Action error: {}", e),
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
    /// Action search paths are automatically discovered from the environment variable
    /// `MORFLOW_ACTIONS_PATH`, binary directory, and target folders.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<MorflowPipeline, MorflowError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_str(&content)
    }

    /// Parses and compiles a `.morf` pipeline string.
    #[allow(clippy::should_implement_trait)]
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

        let registry = Arc::new(ActionRegistry::default());
        let pipeline = MorflowPipeline { ast, registry };
        pipeline.preload_actions()?;
        Ok(pipeline)
    }
}

/// An executable Morflow pipeline instance with auto-parallelization scheduling.
#[derive(Clone)]
pub struct MorflowPipeline {
    pub ast: Pipeline,
    pub registry: Arc<ActionRegistry>,
}

impl MorflowPipeline {
    /// Returns the pipeline parameter declarations (`accept $param`).
    pub fn params(&self) -> &[PipelineParam] {
        &self.ast.params
    }

    /// Preloads all actions declared across all steps in this pipeline into memory.
    fn preload_actions(&self) -> Result<(), MorflowError> {
        let resolver = crate::resolver::ActionResolver::from_imports(&self.ast.imports);
        let action_names = collect_action_names(&self.ast.statements);
        for action in action_names {
            let (target_pack, real_action_name) = resolver.resolve(&action);
            if let Some(pack) = target_pack {
                self.registry
                    .get_or_load_in_pack(&pack, &real_action_name)
                    .map_err(MorflowError::Action)?;
            } else {
                self.registry
                    .get_or_load(&action)
                    .map_err(MorflowError::Action)?;
            }
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
        let resolver = crate::resolver::ActionResolver::from_imports(&self.ast.imports);
        let scheduler = AutoParallelScheduler::new(Arc::clone(&self.registry), resolver);
        scheduler.execute(&self.ast.statements, env)
    }
}

pub fn collect_action_names(statements: &[Statement]) -> Vec<String> {
    let mut names = Vec::new();
    collect_actions_internal(statements, &mut names);
    names
}

fn collect_actions_internal(statements: &[Statement], names: &mut Vec<String>) {
    for stmt in statements {
        let Statement::Flow(chain) = stmt;
        for step in &chain.steps {
            match step {
                FlowStep::Action(call) => {
                    if call.name != "emit" && call.name != "resurface" && !names.contains(&call.name) {
                        names.push(call.name.clone());
                    }
                }
                FlowStep::Each(each_loop) => {
                    collect_actions_internal(&each_loop.body, names);
                }
                FlowStep::IfElse(if_else) => {
                    collect_actions_internal(&if_else.then_branch, names);
                    if let Some(else_branch) = &if_else.else_branch {
                        collect_actions_internal(else_branch, names);
                    }
                }
                FlowStep::Route(route) => {
                    for arm in &route.arms {
                        collect_actions_internal(&arm.body, names);
                    }
                    if let Some(default_arm) = &route.default_arm {
                        collect_actions_internal(default_arm, names);
                    }
                }
                _ => {}
            }
        }
    }
}
