use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use core_types::Payload;
use parser::ast::{FlowStep, Pipeline, PipelineParam, Statement};

use crate::outputs::PipelineOutputs;
use crate::registry::{ActionRegistry, LoadedAction};
use crate::scheduler::AutoParallelScheduler;
use crate::types::{coerce_host_payload, default_payload, ptype_of};

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
                errs.iter()
                    .map(parser::format_error)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        // Perform strict static compile-time validation (SSA / emit rules)
        crate::validator::validate_pipeline(&ast)?;

        let registry = Arc::new(ActionRegistry::default());
        let mut pipeline = MorflowPipeline {
            ast,
            registry,
            actions: HashMap::new(),
        };
        pipeline.preload_actions()?;
        Ok(pipeline)
    }
}

/// An executable Morflow pipeline instance with auto-parallelization scheduling.
#[derive(Clone)]
pub struct MorflowPipeline {
    pub ast: Pipeline,
    pub registry: Arc<ActionRegistry>,
    pub actions: HashMap<String, Arc<LoadedAction>>,
}

impl MorflowPipeline {
    /// Returns the pipeline parameter declarations (`accept $param`).
    pub fn params(&self) -> &[PipelineParam] {
        &self.ast.params
    }

    /// Preloads all actions declared across all steps in this pipeline into memory.
    fn preload_actions(&mut self) -> Result<(), MorflowError> {
        let resolver = crate::resolver::ActionResolver::from_imports(&self.ast.imports)
            .map_err(MorflowError::Compile)?;
        for name in collect_action_names(&self.ast.statements) {
            let identity = resolver
                .resolve(&name, |p, v| self.registry.catalog(p, v))
                .map_err(MorflowError::Action)?;
            let action = self
                .registry
                .get_or_load(&identity)
                .map_err(MorflowError::Action)?;
            self.actions.insert(name, action);
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
    ///
    /// Each supplied payload is checked against the type its `accept` declared,
    /// and a parameter with no supplied value falls back to its default, again
    /// in the representation its declared type calls for.
    pub fn run_args(&mut self, args: Vec<Payload>) -> Result<PipelineOutputs, MorflowError> {
        let mut env: HashMap<String, Payload> = HashMap::new();

        // Bind input arguments to pipeline parameters
        for (i, param) in self.ast.params.iter().enumerate() {
            let declared = ptype_of(&param.param_type);
            let bound = match args.get(i) {
                Some(supplied) => Some(coerce_host_payload(&declared, supplied.clone())),
                None => default_payload(param),
            };
            let Some(bound) = bound else {
                continue;
            };
            declared
                .verify_payload(&bound)
                .map_err(|e| MorflowError::TypeMismatch(format!("${}: {}", param.name, e)))?;
            env.insert(param.name.clone(), bound);
        }

        // If pipeline has no declared parameters but an argument was passed, bind it to $input
        if self.ast.params.is_empty() && !args.is_empty() {
            env.insert("input".to_string(), args[0].clone());
        }

        // Execute via auto-parallel scheduler
        let scheduler = AutoParallelScheduler::new(self.actions.clone());
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
                    if call.name != "emit" && !names.contains(&call.name) {
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

#[cfg(test)]
mod version_tests {
    use super::*;
    use crate::artifact::*;
    use std::fs;

    fn publish(root: &Path, version: &str, concrete: &str, library: &str) {
        let id = ActionIdentity::new("fixture", version, "transform").unwrap();
        let deps = std::env::current_exe().unwrap();
        let debug = deps.parent().unwrap().parent().unwrap();
        let prefix = if cfg!(windows) { "" } else { "lib" };
        let bytes = fs::read(debug.join(format!(
            "{prefix}{library}.{}",
            std::env::consts::DLL_EXTENSION
        )))
        .expect("Build native action fixtures first with cargo build --workspace");
        let receipt = ArtifactReceipt {
            identity: id.clone(),
            concrete_version: concrete.into(),
            repository: "test/fixture".into(),
            sha256: digest(&bytes),
        };
        let _guard = CacheGuard::acquire(root, &id.to_string()).unwrap();
        atomic_write(&id.path(root), &bytes).unwrap();
        atomic_write(
            &receipt_path(&id.path(root)),
            &serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
    }
    fn pipeline(root: &Path, source: &str) -> MorflowPipeline {
        let mut pipeline = MorflowPipeline {
            ast: parser::parse(source).unwrap(),
            registry: Arc::new(ActionRegistry::new(vec![root.into()])),
            actions: HashMap::new(),
        };
        pipeline.preload_actions().unwrap();
        pipeline
    }
    fn output(p: &mut MorflowPipeline) -> f32 {
        p.run(Payload::Tensor(core_types::Tensor::from_f32_slice(&[3.0])))
            .unwrap()
            .into_single()
            .unwrap()
            .as_tensor()
            .unwrap()
            .to_vec_f32()[0]
    }
    #[test]
    fn reloading_latest_in_same_process_preserves_existing_pipeline() {
        let root = tempfile::tempdir().unwrap();
        publish(root.path(), "latest", "0.1.0", "identity");
        let source =
            "from fixture/latest import transform\naccept Tensor x\n$x >> transform >> emit";
        let mut old = pipeline(root.path(), source);
        assert_eq!(output(&mut old), 3.0);
        publish(root.path(), "latest", "0.2.0", "neg");
        let mut new = pipeline(root.path(), source);
        assert_eq!(output(&mut old), 3.0);
        assert_eq!(output(&mut new), -3.0);
        assert_eq!(new.actions["transform"].receipt.concrete_version, "0.2.0");
    }
    #[test]
    fn two_aliased_versions_do_not_collide() {
        let root = tempfile::tempdir().unwrap();
        publish(root.path(), "0.1.0", "0.1.0", "identity");
        publish(root.path(), "0.2.0", "0.2.0", "neg");
        let source = "import fixture/0.1.0 as old\nimport fixture/0.2.0 as new\naccept Tensor x\n$x >> old.transform >> emit(\"old\")\n$x >> new.transform >> emit(\"new\")";
        let mut p = pipeline(root.path(), source);
        let out = p
            .run(Payload::Tensor(core_types::Tensor::from_f32_slice(&[3.0])))
            .unwrap();
        assert_eq!(out["old"].as_tensor().unwrap().to_vec_f32(), vec![3.0]);
        assert_eq!(out["new"].as_tensor().unwrap().to_vec_f32(), vec![-3.0]);
    }
    #[test]
    fn offline_checker_uses_the_same_exact_artifact_as_execution() {
        let root = tempfile::tempdir().unwrap();
        publish(root.path(), "0.1.0", "0.1.0", "identity");
        publish(root.path(), "0.2.0", "0.2.0", "neg");
        publish(root.path(), "latest", "0.1.0", "identity");
        let source =
            "from fixture/0.2.0 import transform\naccept Tensor x\n$x >> transform >> emit";
        let path = root.path().join("pipeline.morf");
        fs::write(&path, source).unwrap();
        crate::check::check_pipeline(&path, Some(root.path())).unwrap();
        assert_eq!(
            output(&mut pipeline(
                root.path(),
                &fs::read_to_string(path).unwrap()
            )),
            -3.0
        );
    }

    #[test]
    fn loader_rejects_unversioned_missing_receipts_and_tampering() {
        let root = tempfile::tempdir().unwrap();
        let id = ActionIdentity::new("fixture", "0.2.0", "transform").unwrap();
        fs::write(root.path().join("transform_action.so"), b"legacy").unwrap();
        let registry = ActionRegistry::new(vec![root.path().into()]);
        assert!(registry.get_or_load(&id).is_err());
        atomic_write(&id.path(root.path()), b"unverified").unwrap();
        assert!(registry.get_or_load(&id).is_err());
        publish(root.path(), "0.2.0", "0.2.0", "identity");
        fs::write(id.path(root.path()), b"tampered").unwrap();
        assert!(registry.get_or_load(&id).is_err());
        publish(root.path(), "0.2.0", "0.2.0", "identity");
        assert!(registry
            .get_or_load(&ActionIdentity::new("fixture", "0.1.0", "transform").unwrap())
            .is_err());
        assert!(registry.get_or_load(&id).is_ok());
    }
}
