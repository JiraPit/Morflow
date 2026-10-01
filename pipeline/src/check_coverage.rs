//! Per-call static coverage, including every possible control-flow branch.
use super::*;
use core_types::{Dimension, InputDescriptor, Shape, ShapeCheckResult};

#[derive(Debug)]
pub struct ActionCoverage {
    pub location: String,
    pub action: String,
    pub input: ValueShape,
    pub output: ValueShape,
    pub status: &'static str,
    pub reason: String,
}
#[derive(Default, Debug)]
pub struct ShapeCoverage {
    pub actions: Vec<ActionCoverage>,
    pub(super) outputs: HashMap<String, Option<PType>>,
}
impl ShapeCoverage {
    pub fn ready(&self) -> usize {
        self.actions.iter().filter(|a| a.status == "Ready").count()
    }
    pub fn percentage(&self) -> Option<f64> {
        (!self.actions.is_empty()).then(|| 100.0 * self.ready() as f64 / self.actions.len() as f64)
    }
    pub fn has_invalid(&self) -> bool {
        self.actions.iter().any(|a| a.status == "Invalid")
    }
    pub fn analyze(
        ast: &Pipeline,
        loaded: &HashMap<String, Arc<LoadedAction>>,
    ) -> Result<Self, String> {
        let mut report = Self::default();
        let mut vars = ast
            .params
            .iter()
            .map(|p| (p.name.clone(), ptype_of(&p.param_type)))
            .collect();
        for (i, Statement::Flow(chain)) in ast.statements.iter().enumerate() {
            report.flow(chain, None, &mut vars, loaded, &format!("Flow {}", i + 1))?;
        }
        Ok(report)
    }
    fn body(
        &mut self,
        body: &[Statement],
        mut current: Option<PType>,
        vars: &mut HashMap<String, PType>,
        loaded: &HashMap<String, Arc<LoadedAction>>,
        path: &str,
    ) -> Result<Option<PType>, String> {
        for (i, Statement::Flow(chain)) in body.iter().enumerate() {
            current = self.flow(
                chain,
                current,
                vars,
                loaded,
                &format!("{path}/flow {}", i + 1),
            )?;
        }
        Ok(current)
    }
    pub(super) fn branch(
        &mut self,
        step: &FlowStep,
        current: Option<PType>,
        vars: &mut HashMap<String, PType>,
        loaded: &HashMap<String, Arc<LoadedAction>>,
        path: &str,
    ) -> Result<Option<PType>, String> {
        let branches: Vec<(&str, &[Statement])> = match step {
            FlowStep::IfElse(b) => vec![
                ("then", &b.then_branch),
                ("else", b.else_branch.as_deref().unwrap_or(&[])),
            ],
            FlowStep::Route(b) => {
                let mut bodies: Vec<_> =
                    b.arms.iter().map(|a| ("arm", a.body.as_slice())).collect();
                bodies.push(("default", b.default_arm.as_deref().unwrap_or(&[])));
                bodies
            }
            _ => unreachable!(),
        };
        let mut results = Vec::new();
        let mut environments = Vec::new();
        for (i, (name, body)) in branches.iter().enumerate() {
            let mut env = vars.clone();
            results.push(self.body(
                body,
                current.clone(),
                &mut env,
                loaded,
                &format!("{path}/{name} {}", i + 1),
            )?);
            environments.push(env);
        }
        // Only variables available on every path remain statically available.
        if let Some(first) = environments.first() {
            let mut merged = HashMap::new();
            for (key, ty) in first {
                if environments.iter().all(|env| env.contains_key(key)) {
                    let value = environments
                        .iter()
                        .skip(1)
                        .fold(ValueShape::from_ptype(ty), |a, env| {
                            join(&a, &ValueShape::from_ptype(&env[key]))
                        });
                    merged.insert(key.clone(), value.to_ptype());
                }
            }
            *vars = merged;
        }
        Ok(results
            .into_iter()
            .reduce(|a, b| match (a, b) {
                (Some(a), Some(b)) => {
                    Some(join(&ValueShape::from_ptype(&a), &ValueShape::from_ptype(&b)).to_ptype())
                }
                _ => None,
            })
            .flatten())
    }
    fn flow(
        &mut self,
        chain: &FlowChain,
        mut current: Option<PType>,
        vars: &mut HashMap<String, PType>,
        loaded: &HashMap<String, Arc<LoadedAction>>,
        path: &str,
    ) -> Result<Option<PType>, String> {
        for (i, step) in chain.steps.iter().enumerate() {
            let location = format!("{path}/step {}", i + 1);
            match step {
                FlowStep::Var(v) => {
                    current = Some(sliced_var_type(
                        vars.get(&v.name).cloned().ok_or_else(|| {
                            format!("{location}: variable '${}' is unavailable", v.name)
                        })?,
                        v,
                    )?)
                }
                FlowStep::Tap(name) => {
                    if let Some(ty) = &current {
                        vars.insert(name.clone(), ty.clone());
                    }
                }
                FlowStep::Action(call) => {
                    // Built-ins route values and have no native shapecheck contract.
                    if call.name == "emit" || call.name == "resurface" {
                        continue;
                    }
                    let mut call = call.clone();
                    if current.is_none() {
                        if let Some(Value::Var(v)) = call.positional_args.first() {
                            current = Some(sliced_var_type(
                                vars.get(&v.name).cloned().ok_or_else(|| {
                                    format!("{location}: variable '${}' is unavailable", v.name)
                                })?,
                                v,
                            )?);
                            call.positional_args.remove(0);
                        }
                    }
                    let input = current
                        .as_ref()
                        .map(ValueShape::from_ptype)
                        .unwrap_or(ValueShape::Unknown);
                    let action = loaded
                        .get(&call.name)
                        .ok_or_else(|| format!("Action '{}' not loaded", call.name))?;
                    // Host arguments are overridable even when a default is declared.
                    let verdict = action.shapecheck(
                        InputDescriptor::partial(input.clone()),
                        call_args(&call, &[]),
                    );
                    let (status, output, reason) = match verdict {
                        ShapeCheckResult::Ready { output, .. } => {
                            ("Ready", output, "Shape contract satisfied".to_string())
                        }
                        ShapeCheckResult::Deferred { output, unresolved } => (
                            "Deferred",
                            output,
                            unresolved
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .join("; "),
                        ),
                        ShapeCheckResult::Invalid { reason } => {
                            ("Invalid", ValueShape::Unknown, reason.to_string())
                        }
                    };
                    current = Some(output.to_ptype());
                    self.actions.push(ActionCoverage {
                        location: location.clone(),
                        action: call.name,
                        input,
                        output,
                        status,
                        reason,
                    });
                }
                FlowStep::Each(loop_) => {
                    let input = current
                        .clone()
                        .ok_or_else(|| format!("{location}: each requires input"))?;
                    let element = input
                        .each_loop_var()
                        .or_else(|e| {
                            if matches!(input.spec(), Some(ShapeSpec::AnyRank)) {
                                Ok(unranked_like(&input))
                            } else {
                                Err(e)
                            }
                        })
                        .map_err(|e| format!("{location}: {e}"))?;
                    let mut env = vars.clone();
                    env.insert(loop_.var_name.clone(), element.clone());
                    let output = self.body(
                        &loop_.body,
                        Some(element.clone()),
                        &mut env,
                        loaded,
                        &format!("{location}/each ${}", loop_.var_name),
                    )?;
                    if let Some(body) = &output {
                        if !same_payload_kind(&element, body)
                            || each_rank_mismatch(&element, body).is_some()
                        {
                            return Err(format!(
                                "{location}: each body must preserve its payload kind and rank"
                            ));
                        }
                    }
                    current = output.map(|body| each_output_type(&input, &body));
                }
                FlowStep::IfElse(_) | FlowStep::Route(_) => {
                    current = self.branch(step, current, vars, loaded, &location)?
                }
            }
            self.outputs.insert(location, current.clone());
        }
        Ok(current)
    }
}

/// Retain only dimensions and ordered components common to every branch.
fn join(a: &ValueShape, b: &ValueShape) -> ValueShape {
    if a == b {
        return a.clone();
    }
    match (a, b) {
        (ValueShape::Composite(a), ValueShape::Composite(b)) if a.len() == b.len() => {
            ValueShape::composite(a.iter().zip(b).map(|(a, b)| join(a, b)))
        }
        (ValueShape::Leaf { kind: a, shape: sa }, ValueShape::Leaf { kind: b, shape: sb })
            if a == b =>
        {
            match (sa.as_ref().into_option(), sb.as_ref().into_option()) {
                (Some(sa), Some(sb)) if sa.rank() == sb.rank() => ValueShape::Leaf {
                    kind: *a,
                    shape: core_types::abi_stable::std_types::ROption::RSome(Shape::new(
                        sa.dims.iter().zip(&sb.dims).map(|(a, b)| {
                            if a == b {
                                *a
                            } else {
                                Dimension::Unknown
                            }
                        }),
                    )),
                },
                _ => ValueShape::unranked(*a),
            }
        }
        _ => ValueShape::Unknown,
    }
}
