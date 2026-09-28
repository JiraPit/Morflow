use std::collections::{HashMap, HashSet};

use parser::ast::{FlowChain, FlowStep, Pipeline, Statement, Value};

use crate::engine::MorflowError;

/// Statically validates that a pipeline adheres to:
/// 1. Strict Single Assignment / No-External-Write rule.
/// 2. Emit rules (terminal step, top-level only, unique names if multiple flows emit).
pub fn validate_pipeline(pipeline: &Pipeline) -> Result<(), MorflowError> {
    let mut writer_history: HashMap<String, Vec<String>> = HashMap::new();
    let mut scope_vars: HashSet<String> = HashSet::new();
    let mut top_level_emits: Vec<Option<String>> = Vec::new();

    // 1. Register declared pipeline parameters
    for param in &pipeline.params {
        scope_vars.insert(param.name.clone());
        writer_history
            .entry(param.name.clone())
            .or_default()
            .push(format!("parameter '${}'", param.name));
    }

    // 2. Validate top-level flows sequentially
    for (stmt_idx, stmt) in pipeline.statements.iter().enumerate() {
        validate_statement(
            stmt,
            stmt_idx + 1,
            &mut scope_vars,
            &mut writer_history,
            &mut top_level_emits,
            false,
            None,
        )?;
    }

    // 3. Validate emit rules across all flows in the pipeline
    if top_level_emits.len() > 1 {
        // When multiple flows emit, every emit call must specify a non-empty name
        let mut seen_names = HashSet::new();
        for name_opt in &top_level_emits {
            match name_opt {
                Some(name) if !name.is_empty() => {
                    if !seen_names.insert(name.clone()) {
                        return Err(MorflowError::Compile(format!(
                            "Duplicate emit name '{}': multiple flows emit with the same name.",
                            name
                        )));
                    }
                }
                _ => {
                    return Err(MorflowError::Compile(
                        "Multiple flows emit outputs, but one or more emit calls are unnamed. When multiple flows emit, each must specify a distinct name (e.g. emit(\"name\")).".to_string(),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn validate_statement(
    stmt: &Statement,
    stmt_num: usize,
    scope_vars: &mut HashSet<String>,
    writer_history: &mut HashMap<String, Vec<String>>,
    top_level_emits: &mut Vec<Option<String>>,
    is_nested_sub_flow: bool,
    parent_desc: Option<&str>,
) -> Result<(), MorflowError> {
    let Statement::Flow(flow) = stmt;
    validate_flow_chain(
        flow,
        stmt_num,
        scope_vars,
        writer_history,
        top_level_emits,
        is_nested_sub_flow,
        parent_desc,
    )
}

fn validate_flow_chain(
    flow: &FlowChain,
    stmt_num: usize,
    scope_vars: &mut HashSet<String>,
    writer_history: &mut HashMap<String, Vec<String>>,
    top_level_emits: &mut Vec<Option<String>>,
    is_nested_sub_flow: bool,
    parent_desc: Option<&str>,
) -> Result<(), MorflowError> {
    let flow_name = get_flow_descriptor(flow, stmt_num, parent_desc);

    for step in &flow.steps {
        match step {
            FlowStep::Action(call) if call.name == "emit" || call.name == "resurface" => {
                if is_nested_sub_flow {
                    return Err(MorflowError::Compile(
                        "'emit' cannot be called inside a nested sub-flow (loops or branches); emit the result from the top-level flow instead.".to_string(),
                    ));
                }

                // Extract name
                let name = if let Some(val) = call.positional_args.first() {
                    match val {
                        Value::String(s) => Some(s.clone()),
                        _ => {
                            return Err(MorflowError::Compile(
                                "emit name argument must be a string (e.g. emit(\"name\"))."
                                    .to_string(),
                            ))
                        }
                    }
                } else if let Some((_, val)) = call.named_args.iter().find(|(k, _)| k == "name") {
                    match val {
                        Value::String(s) => Some(s.clone()),
                        _ => {
                            return Err(MorflowError::Compile(
                                "emit name argument must be a string (e.g. emit(name=\"name\"))."
                                    .to_string(),
                            ))
                        }
                    }
                } else {
                    None
                };

                top_level_emits.push(name);
            }
            FlowStep::Tap(var_name) => {
                let history = writer_history.entry(var_name.clone()).or_default();
                history.push(flow_name.clone());

                if history.len() > 1 || (is_nested_sub_flow && scope_vars.contains(var_name)) {
                    return Err(MorflowError::Compile(format!(
                        "${} is written to by multiple flows: {}.",
                        var_name,
                        history.join(", ")
                    )));
                }
                scope_vars.insert(var_name.clone());
            }
            FlowStep::Each(each_loop) => {
                if scope_vars.contains(&each_loop.var_name) {
                    return Err(MorflowError::Compile(format!(
                        "Loop variable '${}' conflicts with an existing variable name.",
                        each_loop.var_name
                    )));
                }

                let mut inner_scope = scope_vars.clone();
                inner_scope.insert(each_loop.var_name.clone());

                let sub_desc = format!("each (${}) sub-flow", each_loop.var_name);
                for (sub_idx, inner_stmt) in each_loop.body.iter().enumerate() {
                    validate_statement(
                        inner_stmt,
                        sub_idx + 1,
                        &mut inner_scope,
                        writer_history,
                        top_level_emits,
                        true,
                        Some(&sub_desc),
                    )?;
                }
            }
            FlowStep::IfElse(branch) => {
                let mut then_scope = scope_vars.clone();
                for (sub_idx, then_stmt) in branch.then_branch.iter().enumerate() {
                    validate_statement(
                        then_stmt,
                        sub_idx + 1,
                        &mut then_scope,
                        writer_history,
                        top_level_emits,
                        true,
                        Some("if branch sub-flow"),
                    )?;
                }

                if let Some(else_branch) = &branch.else_branch {
                    let mut else_scope = scope_vars.clone();
                    for (sub_idx, else_stmt) in else_branch.iter().enumerate() {
                        validate_statement(
                            else_stmt,
                            sub_idx + 1,
                            &mut else_scope,
                            writer_history,
                            top_level_emits,
                            true,
                            Some("else branch sub-flow"),
                        )?;
                    }
                }
            }
            FlowStep::Route(route) => {
                for arm in &route.arms {
                    let mut arm_scope = scope_vars.clone();
                    for (sub_idx, arm_stmt) in arm.body.iter().enumerate() {
                        validate_statement(
                            arm_stmt,
                            sub_idx + 1,
                            &mut arm_scope,
                            writer_history,
                            top_level_emits,
                            true,
                            Some("route arm sub-flow"),
                        )?;
                    }
                }
                if let Some(default_arm) = &route.default_arm {
                    let mut def_scope = scope_vars.clone();
                    for (sub_idx, def_stmt) in default_arm.iter().enumerate() {
                        validate_statement(
                            def_stmt,
                            sub_idx + 1,
                            &mut def_scope,
                            writer_history,
                            top_level_emits,
                            true,
                            Some("route else arm sub-flow"),
                        )?;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Generates a human-readable identifier for a flow or sub-flow.
fn get_flow_descriptor(flow: &FlowChain, stmt_num: usize, parent_desc: Option<&str>) -> String {
    let mut actions = Vec::new();
    for step in &flow.steps {
        if let FlowStep::Action(call) = step {
            actions.push(call.name.clone());
        }
    }

    if let Some(parent) = parent_desc {
        if !actions.is_empty() {
            format!("{} ({})", parent, actions.join(" >> "))
        } else {
            parent.to_string()
        }
    } else if !actions.is_empty() {
        actions.join(" >> ")
    } else {
        format!("flow {}", stmt_num)
    }
}
