use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ariadne::{Color, Label, Report, ReportKind, Source};
use core_types::DataType;
use parser::ast::*;
use rich_rust::prelude::*;
use rich_rust::r#box::ROUNDED;

use crate::cli::{get_host_platform, resolve_action_cache_dir, resolve_repo};
use crate::engine::collect_action_names;
use crate::registry::{ActionRegistry, LoadedAction};
use crate::resolver::ActionResolver;
use crate::validator::validate_pipeline;

/// Finds byte offsets for an identifier in source text, ignoring comments
fn find_token_span(source: &str, token: &str, occurrence: usize) -> std::ops::Range<usize> {
    let mut count = 0;
    let mut offset = 0;
    for line in source.lines() {
        let trimmed = line.trim();
        let is_comment = trimmed.starts_with('#') || trimmed.starts_with("//");
        if !is_comment {
            let mut line_offset = 0;
            while let Some(pos) = line[line_offset..].find(token) {
                let abs_pos = offset + line_offset + pos;
                let before = if abs_pos > 0 {
                    source[..abs_pos].chars().last()
                } else {
                    None
                };
                let after = source[abs_pos + token.len()..].chars().next();
                let is_ident_char = |c: Option<char>| {
                    c.map(|ch| ch.is_alphanumeric() || ch == '_')
                        .unwrap_or(false)
                };

                let valid_boundary = if token.starts_with('$') {
                    !is_ident_char(after)
                } else {
                    !is_ident_char(before) && !is_ident_char(after)
                };

                if valid_boundary {
                    if count == occurrence {
                        return abs_pos..abs_pos + token.len();
                    }
                    count += 1;
                }
                line_offset += pos + token.len();
            }
        }
        offset += line.len() + 1;
    }

    // Fallback: search anywhere
    if let Some(pos) = source.find(token) {
        pos..pos + token.len()
    } else {
        0..source.len().min(1)
    }
}

/// Checks compatibility between output of previous step and required input of next action
fn are_types_compatible(source_type: DataType, target_input_type: DataType) -> bool {
    if source_type == target_input_type {
        return true;
    }
    // Universal converters accepting raw bytes
    if target_input_type == DataType::RawBytes {
        return true;
    }
    match (source_type, target_input_type) {
        // Disallow direct piping between Audio and Image without conversion
        (DataType::Audio, DataType::Image) => false,
        (DataType::Image, DataType::Audio) => false,
        // Audio and Image can bridge to/from Tensor
        (DataType::Audio, DataType::Tensor) => true,
        (DataType::Image, DataType::Tensor) => true,
        (DataType::Tensor, DataType::Audio) => true,
        (DataType::Tensor, DataType::Image) => true,
        (DataType::Composite, DataType::Tensor) => true,
        (DataType::Tensor, DataType::Composite) => true,
        _ => false,
    }
}

fn format_data_type(dt: DataType) -> &'static str {
    match dt {
        DataType::RawBytes => "RawBytes",
        DataType::Tensor => "Tensor",
        DataType::Composite => "Composite",
        DataType::Image => "Image",
        DataType::Audio => "Audio",
    }
}

/// Executes the comprehensive static check & dry run on a .morf file
pub fn check_pipeline(
    file_path: &Path,
    custom_cache_path: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let console = Console::new();

    if !file_path.exists() {
        console.print(&format!(
            "[bold red]Error:[/] Pipeline file '{}' does not exist.",
            file_path.display()
        ));
        return Err(format!("File '{}' not found", file_path.display()).into());
    }

    let source = fs::read_to_string(file_path)?;
    let filename_str = file_path.display().to_string();

    // ==========================================
    // Phase 1: Syntax & Grammar Check (Chumsky + Ariadne)
    // ==========================================
    let ast = match parser::parse(&source) {
        Ok(p) => p,
        Err(errs) => {
            console.rule(Some("Morflow Syntax Verification"));
            for err in &errs {
                let span = err.span();
                let msg = match err.reason() {
                    chumsky::error::SimpleReason::Unexpected => {
                        let found_str = match err.found() {
                            Some(c) => format!("character '{}'", c),
                            None => "end of input".to_string(),
                        };
                        format!("Unexpected {}", found_str)
                    }
                    chumsky::error::SimpleReason::Unclosed { delimiter, .. } => {
                        format!("Unclosed delimiter '{}'", delimiter)
                    }
                    chumsky::error::SimpleReason::Custom(s) => s.clone(),
                };

                let expected_tokens: Vec<String> = err
                    .expected()
                    .map(|c| match c {
                        Some(ch) => format!("'{}'", ch),
                        None => "end of input".to_string(),
                    })
                    .collect();

                let mut report =
                    Report::build(ReportKind::Error, (filename_str.as_str(), span.clone()))
                        .with_code("E001")
                        .with_message(format!("Syntax error: {}", msg))
                        .with_label(
                            Label::new((filename_str.as_str(), span))
                                .with_message(&msg)
                                .with_color(Color::Red),
                        );

                if !expected_tokens.is_empty() {
                    report = report.with_help(format!("Expected: {}", expected_tokens.join(", ")));
                }

                let _ = report
                    .finish()
                    .print((filename_str.as_str(), Source::from(&source)));
            }

            console.print("");
            console.print(&format!(
                "[bold red]✗ Check failed:[/] Found {} syntax error(s) in [dim]{}[/].",
                errs.len(),
                file_path.display()
            ));
            return Err("Syntax validation failed".into());
        }
    };

    // ==========================================
    // Phase 2: Semantic & Flow Validation (SSA / Emits)
    // ==========================================
    if let Err(err) = validate_pipeline(&ast) {
        console.rule(Some("Morflow Semantic & SSA Verification"));
        let err_str = err.to_string();

        let token_target = if let Some(dollar_idx) = err_str.find('$') {
            let var_slice = &err_str[dollar_idx..];
            let end_var = var_slice
                .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '$')
                .unwrap_or(var_slice.len());
            &var_slice[..end_var]
        } else if err_str.contains("'emit'") {
            "emit"
        } else {
            ""
        };

        let span = if !token_target.is_empty() {
            find_token_span(&source, token_target, 0)
        } else {
            0..source.len().min(1)
        };

        let report = Report::build(ReportKind::Error, (filename_str.as_str(), span.clone()))
            .with_code("E002")
            .with_message(&err_str)
            .with_label(
                Label::new((filename_str.as_str(), span))
                    .with_message(&err_str)
                    .with_color(Color::Red),
            )
            .with_help(
                "Morflow pipelines enforce strict Single Static Assignment (SSA) and emit rules.",
            );

        let _ = report
            .finish()
            .print((filename_str.as_str(), Source::from(&source)));

        console.print("");
        console.print(&format!(
            "[bold red]✗ Check failed:[/] Semantic error in [dim]{}[/]: {}",
            file_path.display(),
            err_str
        ));
        return Err("Semantic validation failed".into());
    }

    // ==========================================
    // Phase 3: Action Resolution & Dynamic Binary Loading
    // ==========================================
    let resolver = ActionResolver::from_imports(&ast.imports);
    let action_names = collect_action_names(&ast.statements);

    let mut search_paths = ActionRegistry::default_search_paths();
    if let Some(custom) = custom_cache_path {
        search_paths.insert(0, custom.to_path_buf());
    }
    let registry = ActionRegistry::new(search_paths);
    let mut loaded_actions: HashMap<String, Arc<LoadedAction>> = HashMap::new();

    for act_name in &action_names {
        if act_name == "emit" || act_name == "resurface" {
            continue;
        }

        let (target_pack, real_act) = resolver.resolve(act_name);
        let loaded_res = if let Some(pack) = &target_pack {
            registry
                .get_or_load_in_pack(pack, &real_act)
                .or_else(|_| registry.get_or_load(&real_act))
        } else {
            registry.get_or_load(&real_act)
        };

        match loaded_res {
            Ok(action) => {
                loaded_actions.insert(act_name.clone(), action);
            }
            Err(_) => {
                let span = find_token_span(&source, act_name, 0);
                let report =
                    Report::build(ReportKind::Error, (filename_str.as_str(), span.clone()))
                        .with_code("E003")
                        .with_message(format!(
                            "Action '{}' is not installed or available locally",
                            act_name
                        ))
                        .with_label(
                            Label::new((filename_str.as_str(), span))
                                .with_message(format!(
                                    "Action binary for '{}' not found in cache or search paths",
                                    act_name
                                ))
                                .with_color(Color::Red),
                        )
                        .with_help(format!(
                            "Run 'morflow prep {}' first to download and prepare required action binaries before running check.",
                            file_path.display()
                        ));

                let _ = report
                    .finish()
                    .print((filename_str.as_str(), Source::from(&source)));

                console.print("");
                console.print(&format!(
                    "[bold red]✗ Check failed:[/] Missing action binary for '[bold]{}[/]'.",
                    act_name
                ));
                console.print(&format!(
                    "[yellow]Tip:[/] Run [bold cyan]morflow prep {}[/] to install required actions.",
                    file_path.display()
                ));
                return Err("Missing action binary".into());
            }
        }
    }

    // ==========================================
    // Phase 4: Dynamic Type Checking Dry-Run (via get_input_type & get_output_type)
    // ==========================================
    let mut var_types: HashMap<String, Option<DataType>> = HashMap::new();

    // Initialize pipeline parameters as unconstrained input payloads
    for param in &ast.params {
        var_types.insert(param.name.clone(), None);
    }

    struct FlowInspection {
        stmt_idx: usize,
        source_desc: String,
        steps_summary: Vec<String>,
        output_desc: String,
        type_transitions: Vec<DataType>,
    }

    let mut inspected_flows: Vec<FlowInspection> = Vec::new();

    for (stmt_idx, stmt) in ast.statements.iter().enumerate() {
        let Statement::Flow(chain) = stmt;
        let mut curr_type: Option<DataType> = None;
        let mut source_desc = "stream".to_string();
        let mut output_desc = "-".to_string();
        let mut steps_summary = Vec::new();
        let mut type_transitions = Vec::new();
        let mut prev_step_desc = String::new();
        let mut initial_var_name: Option<String> = None;

        for (step_idx, step) in chain.steps.iter().enumerate() {
            match step {
                FlowStep::Var(var_ref) => {
                    if !var_types.contains_key(&var_ref.name) {
                        let span = find_token_span(&source, &format!("${}", var_ref.name), 0);
                        let report =
                            Report::build(ReportKind::Error, (filename_str.as_str(), span.clone()))
                                .with_code("E004")
                                .with_message(format!(
                                    "Undefined variable access: '${}'",
                                    var_ref.name
                                ))
                                .with_label(
                                    Label::new((filename_str.as_str(), span))
                                        .with_message(format!(
                                            "Variable '${}' is read before being declared or assigned",
                                            var_ref.name
                                        ))
                                        .with_color(Color::Red),
                                )
                                .with_help(format!(
                                    "Declare '${}' as a parameter with 'accept ${}' or tap it from an earlier flow.",
                                    var_ref.name, var_ref.name
                                ));

                        let _ = report
                            .finish()
                            .print((filename_str.as_str(), Source::from(&source)));

                        console.print("");
                        console.print(&format!(
                            "[bold red]✗ Check failed:[/] Undefined variable '${}' in [dim]{}[/].",
                            var_ref.name,
                            file_path.display()
                        ));
                        return Err("Undefined variable error".into());
                    }

                    curr_type = *var_types.get(&var_ref.name).unwrap_or(&None);
                    if step_idx == 0 {
                        source_desc = format!("${}", var_ref.name);
                        initial_var_name = Some(var_ref.name.clone());
                    }
                    if let Some(t) = curr_type {
                        type_transitions.push(t);
                    }
                    prev_step_desc = format!("${}", var_ref.name);
                }
                FlowStep::Action(call) => {
                    if call.name == "emit" || call.name == "resurface" {
                        steps_summary.push(call.name.clone());
                        output_desc = "emit".to_string();
                        continue;
                    }

                    let loaded = match loaded_actions.get(&call.name) {
                        Some(a) => a,
                        None => {
                            return Err(format!("Action '{}' not loaded", call.name).into());
                        }
                    };

                    // Retrieve input and output types directly from the loaded action binary!
                    let action_in = loaded.input_type;
                    let action_out = loaded.output_type;

                    if let Some(src_type) = curr_type {
                        if !are_types_compatible(src_type, action_in) {
                            let curr_span = find_token_span(&source, &call.name, 0);
                            let prev_span = if !prev_step_desc.is_empty() {
                                find_token_span(&source, &prev_step_desc, 0)
                            } else {
                                curr_span.clone()
                            };

                            let report = Report::build(
                                ReportKind::Error,
                                (filename_str.as_str(), curr_span.clone()),
                            )
                            .with_code("E005")
                            .with_message(format!(
                                "Data type mismatch: '{}' requires input type {:?}, but previous step produces {:?}",
                                call.name, action_in, src_type
                            ))
                            .with_label(
                                Label::new((filename_str.as_str(), prev_span))
                                    .with_message(format!("Produces {:?}", src_type))
                                    .with_color(Color::Blue),
                            )
                            .with_label(
                                Label::new((filename_str.as_str(), curr_span))
                                    .with_message(format!(
                                        "Action '{}' get_input_type() returned {:?}",
                                        call.name, action_in
                                    ))
                                    .with_color(Color::Red),
                            )
                            .with_help("Insert an explicit conversion action (e.g. 'to_tensor') or adjust the pipeline routing.");

                            let _ = report
                                .finish()
                                .print((filename_str.as_str(), Source::from(&source)));

                            console.print("");
                            console.print(&format!(
                                "[bold red]✗ Check failed:[/] Type mismatch at step '{}' in [dim]{}[/].",
                                call.name,
                                file_path.display()
                            ));
                            return Err("Type mismatch error".into());
                        }
                    } else if let Some(init_var) = &initial_var_name {
                        // Infer the initial variable's accepted type from the first action it is fed into
                        var_types.insert(init_var.clone(), Some(action_in));
                    }

                    curr_type = Some(action_out);
                    steps_summary.push(call.name.clone());
                    type_transitions.push(action_out);
                    prev_step_desc = call.name.clone();
                }
                FlowStep::Tap(var_name) => {
                    var_types.insert(var_name.clone(), curr_type);
                    steps_summary.push(format!(">> ${}", var_name));
                    output_desc = format!("${}", var_name);
                    prev_step_desc = format!("${}", var_name);
                }
                FlowStep::Each(each_loop) => {
                    var_types.insert(each_loop.var_name.clone(), curr_type);
                    steps_summary.push(format!("each (${})", each_loop.var_name));
                }
                FlowStep::IfElse(_) => {
                    steps_summary.push("if/else".to_string());
                }
                FlowStep::Route(_) => {
                    steps_summary.push("route".to_string());
                }
            }
        }

        inspected_flows.push(FlowInspection {
            stmt_idx: stmt_idx + 1,
            source_desc,
            steps_summary,
            output_desc,
            type_transitions,
        });
    }

    // ==========================================
    // Phase 5: Rich Presentation (rich_rust)
    // ==========================================
    console.rule(Some("Morflow Pipeline Check & Dry-Run"));
    console.print(&format!(
        "  [bold cyan]Pipeline:[/]        [green]{}[/]",
        file_path.display()
    ));

    let (platform, ext) = get_host_platform();
    let cache_dir = resolve_action_cache_dir(custom_cache_path.map(|p| p.to_path_buf()));
    let repo = resolve_repo();

    console.print(&format!(
        "  [bold cyan]Host Platform:[/]   [yellow]{}[/] (.{})",
        platform, ext
    ));
    console.print(&format!(
        "  [bold cyan]Cache Directory:[/] [dim]{}[/]",
        cache_dir.display()
    ));
    console.print(&format!(
        "  [bold cyan]Repository:[/]      [blue]{}[/]",
        repo
    ));

    if !ast.params.is_empty() {
        let params_str = ast
            .params
            .iter()
            .map(|p| {
                let ty = var_types
                    .get(&p.name)
                    .and_then(|t| *t)
                    .map(format_data_type)
                    .unwrap_or("Payload (Any)");
                if let Some(def) = &p.default_value {
                    format!("${}: {} = {:?}", p.name, ty, def)
                } else {
                    format!("${}: {}", p.name, ty)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        console.print(&format!(
            "  [bold cyan]Parameters:[/]      [dim]{}[/]",
            params_str
        ));
    }
    console.print("");

    // 1. Flow Execution Simulation Table
    let mut flow_table = Table::new()
        .box_style(&ROUNDED)
        .border_style(Style::parse("bright_cyan").unwrap_or_default())
        .header_style(Style::parse("bold white on blue").unwrap_or_default())
        .with_column(Column::new("Flow").width(6).justify(JustifyMethod::Center))
        .with_column(Column::new("Input Source").width(16).no_wrap())
        .with_column(Column::new("Execution Chain").no_wrap())
        .with_column(Column::new("Output Destination").width(18).no_wrap())
        .with_column(Column::new("Type Flow").no_wrap());

    for flow in &inspected_flows {
        let chain_str = flow.steps_summary.join(" [cyan]>>[/] ");
        let type_flow_str = flow
            .type_transitions
            .iter()
            .map(|t| format!("[green]{}[/]", format_data_type(*t)))
            .collect::<Vec<_>>()
            .join(" [dim]→[/] ");

        flow_table.add_row_markup([
            format!("#{}", flow.stmt_idx).as_str(),
            flow.source_desc.as_str(),
            chain_str.as_str(),
            flow.output_desc.as_str(),
            type_flow_str.as_str(),
        ]);
    }

    console.print_renderable(&flow_table);
    console.print("");

    // 2. Action Dynamic Inspection Table (showing actual types from action binaries)
    if !action_names.is_empty() {
        let width = console.width();
        let show_path = width >= 95;

        let mut dep_table = Table::new()
            .box_style(&ROUNDED)
            .border_style(Style::parse("bright_cyan").unwrap_or_default())
            .header_style(Style::parse("bold white on blue").unwrap_or_default())
            .with_column(Column::new("Action").no_wrap())
            .with_column(
                Column::new("get_input_type()")
                    .width(18)
                    .justify(JustifyMethod::Center),
            )
            .with_column(
                Column::new("get_output_type()")
                    .width(18)
                    .justify(JustifyMethod::Center),
            );

        if show_path {
            dep_table = dep_table
                .with_column(Column::new("Binary Origin").no_wrap())
                .with_column(
                    Column::new("Status")
                        .width(14)
                        .justify(JustifyMethod::Center),
                );
        } else {
            dep_table = dep_table.with_column(
                Column::new("Status")
                    .width(14)
                    .justify(JustifyMethod::Center),
            );
        }

        for action_name in &action_names {
            if action_name == "emit" || action_name == "resurface" {
                continue;
            }

            if let Some(loaded) = loaded_actions.get(action_name) {
                let in_type = format_data_type(loaded.input_type);
                let out_type = format_data_type(loaded.output_type);
                let path_display = loaded.path.display().to_string();

                if show_path {
                    dep_table.add_row_markup([
                        action_name.as_str(),
                        format!("[cyan]{}[/]", in_type).as_str(),
                        format!("[green]{}[/]", out_type).as_str(),
                        path_display.as_str(),
                        "[bold green]✓ Verified[/]",
                    ]);
                } else {
                    dep_table.add_row_markup([
                        action_name.as_str(),
                        format!("[cyan]{}[/]", in_type).as_str(),
                        format!("[green]{}[/]", out_type).as_str(),
                        "[bold green]✓ Verified[/]",
                    ]);
                }
            }
        }

        console.print_renderable(&dep_table);
        console.print("");
    }

    console.print(&format!(
        "[bold green]✓ Pipeline Verified:[/] [bold]{}[/] passed all syntax, SSA variable scope, native action FFI, and data type checks.",
        file_path.display()
    ));

    Ok(())
}
