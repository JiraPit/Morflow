use std::collections::{HashMap, HashSet};
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
use crate::scheduler::{extract_dependencies, extract_writes};
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

/// Checks compatibility between output of previous step (source_type)
/// and required/accepted input of next action (target_input_type).
fn are_types_compatible(source_type: DataType, target_input_type: DataType) -> bool {
    // 1. Direct type intersection: does target accept this source type?
    // e.g. target accepts (RawBytes | Tensor), source is RawBytes -> true
    // e.g. target accepts (Tensor | Image | Audio), source is Audio -> true
    // e.g. target accepts Audio, source is Audio -> true
    if target_input_type.intersects(source_type) {
        return true;
    }

    // 2. Composite tensor unpacking:
    if (source_type == DataType::Composite && target_input_type.intersects(DataType::Tensor))
        || (source_type == DataType::Tensor && target_input_type.intersects(DataType::Composite))
    {
        return true;
    }

    false
}

fn format_data_type(dt: DataType) -> String {
    dt.to_string()
}

fn format_var_ref(var_ref: &VarRef) -> String {
    let mut s = format!("${}", var_ref.name);
    if let Some(f) = &var_ref.field {
        s.push('.');
        s.push_str(f);
    }
    if !var_ref.slices.is_empty() {
        s.push('[');
        let slices_str: Vec<String> = var_ref
            .slices
            .iter()
            .map(|slice| match slice {
                SliceItem::Index(idx) => idx.to_string(),
                SliceItem::Full => ":".to_string(),
                SliceItem::Range { start, end, step } => {
                    let mut r = String::new();
                    if let Some(st) = start {
                        r.push_str(&st.to_string());
                    }
                    r.push(':');
                    if let Some(en) = end {
                        r.push_str(&en.to_string());
                    }
                    if let Some(sp) = step {
                        r.push(':');
                        r.push_str(&sp.to_string());
                    }
                    r
                }
                SliceItem::NamedDim { dim_name, index } => format!("{}={}", dim_name, index),
            })
            .collect();
        s.push_str(&slices_str.join(", "));
        s.push(']');
    }
    s
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
    }

    let mut inspected_flows: Vec<FlowInspection> = Vec::new();

    for (stmt_idx, stmt) in ast.statements.iter().enumerate() {
        let Statement::Flow(chain) = stmt;
        let mut curr_type: Option<DataType> = None;
        let mut source_desc = "stream".to_string();
        let mut output_desc = "-".to_string();
        let mut steps_summary = Vec::new();
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
                        source_desc = format_var_ref(var_ref);
                        initial_var_name = Some(var_ref.name.clone());
                    }
                    prev_step_desc = format!("${}", var_ref.name);
                }
                FlowStep::Action(call) => {
                    if call.name == "emit" || call.name == "resurface" {
                        let emit_desc = if let Some(Value::String(s)) = call.positional_args.first() {
                            format!("{}(\"{}\")", call.name, s)
                        } else if let Some((_, Value::String(s))) =
                            call.named_args.iter().find(|(k, _)| k == "name")
                        {
                            format!("{}(\"{}\")", call.name, s)
                        } else {
                            call.name.clone()
                        };
                        steps_summary.push(emit_desc.clone());
                        output_desc = emit_desc;
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
                    prev_step_desc = call.name.clone();
                }
                FlowStep::Tap(var_name) => {
                    var_types.insert(var_name.clone(), curr_type);
                    steps_summary.push(format!("${}", var_name));
                    output_desc = format!("${}", var_name);
                    prev_step_desc = format!("${}", var_name);
                }
                FlowStep::Each(each_loop) => {
                    var_types.insert(each_loop.var_name.clone(), curr_type);
                    let mut inner_type = curr_type;
                    let mut inner_actions = Vec::new();
                    let mut prev_inner_desc = format!("${}", each_loop.var_name);

                    for inner_stmt in &each_loop.body {
                        let Statement::Flow(inner_chain) = inner_stmt;
                        for sub_step in &inner_chain.steps {
                            match sub_step {
                                FlowStep::Var(v) => {
                                    inner_type = *var_types.get(&v.name).unwrap_or(&None);
                                    prev_inner_desc = format!("${}", v.name);
                                }
                                FlowStep::Action(call) => {
                                    if call.name == "emit" || call.name == "resurface" {
                                        continue;
                                    }
                                    inner_actions.push(call.name.clone());
                                    if let Some(loaded) = loaded_actions.get(&call.name) {
                                        let action_in = loaded.input_type;
                                        let action_out = loaded.output_type;

                                        if let Some(src_ty) = inner_type {
                                            if !are_types_compatible(src_ty, action_in) {
                                                let curr_span = find_token_span(&source, &call.name, 0);
                                                let prev_span = find_token_span(&source, &prev_inner_desc, 0);

                                                let report = Report::build(
                                                    ReportKind::Error,
                                                    (filename_str.as_str(), curr_span.clone()),
                                                )
                                                .with_code("E005")
                                                .with_message(format!(
                                                    "Data type mismatch inside each loop: '{}' requires input type {:?}, but previous step produces {:?}",
                                                    call.name, action_in, src_ty
                                                ))
                                                .with_label(
                                                    Label::new((filename_str.as_str(), prev_span))
                                                        .with_message(format!("Produces {:?}", src_ty))
                                                        .with_color(Color::Blue),
                                                )
                                                .with_label(
                                                    Label::new((filename_str.as_str(), curr_span))
                                                        .with_message(format!(
                                                            "Action '{}' get_input_type() returned {:?}",
                                                            call.name, action_in
                                                        ))
                                                        .with_color(Color::Red),
                                                );

                                                let _ = report
                                                    .finish()
                                                    .print((filename_str.as_str(), Source::from(&source)));

                                                return Err("Type mismatch in loop".into());
                                            }
                                        }
                                        inner_type = Some(action_out);
                                        prev_inner_desc = call.name.clone();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    if inner_actions.is_empty() {
                        steps_summary.push(format!("each (${})", each_loop.var_name));
                    } else {
                        steps_summary.push(format!(
                            "each (${}) [⚡ Rayon Parallel] {{ {} }}",
                            each_loop.var_name,
                            inner_actions.join(" >> ")
                        ));
                    }
                    if inner_type.is_some() {
                        curr_type = inner_type;
                    }
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
                    .unwrap_or_else(|| "Payload (Any)".to_string());
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

    // ==========================================
    // Phase 4.5: DAG Dependency & Stage Wave Analysis
    // ==========================================
    let num_flows = ast.statements.len();
    let mut all_flow_reads: Vec<HashSet<String>> = Vec::with_capacity(num_flows);
    let mut all_flow_writes: Vec<HashSet<String>> = Vec::with_capacity(num_flows);
    let mut all_flow_deps: Vec<HashSet<usize>> = vec![HashSet::new(); num_flows];
    let mut all_flow_dep_vars: Vec<HashMap<usize, Vec<String>>> = vec![HashMap::new(); num_flows];
    let mut var_producer: HashMap<String, usize> = HashMap::new();

    for (stmt_idx, stmt) in ast.statements.iter().enumerate() {
        let reads = extract_dependencies(stmt);
        let writes = extract_writes(stmt);

        for var in &reads {
            if let Some(&producer_idx) = var_producer.get(var) {
                if producer_idx != stmt_idx {
                    all_flow_deps[stmt_idx].insert(producer_idx);
                    let vars_vec = all_flow_dep_vars[stmt_idx]
                        .entry(producer_idx)
                        .or_default();
                    if !vars_vec.contains(var) {
                        vars_vec.push(var.clone());
                    }
                }
            }
        }

        for var in &writes {
            var_producer.insert(var.clone(), stmt_idx);
        }

        all_flow_reads.push(reads);
        all_flow_writes.push(writes);
    }

    // Topological Stage Wave Scheduling (matches Rayon AutoParallelScheduler dynamic dispatch)
    let mut remaining: HashSet<usize> = (0..num_flows).collect();
    let mut stages: Vec<Vec<usize>> = Vec::new();
    let mut completed: HashSet<usize> = HashSet::new();

    while !remaining.is_empty() {
        let mut ready: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&idx| all_flow_deps[idx].iter().all(|d| completed.contains(d)))
            .collect();

        if ready.is_empty() {
            // Cycle or unresolvable dependency fallback
            ready = remaining.into_iter().collect();
            ready.sort();
            stages.push(ready);
            break;
        }

        ready.sort();
        for &idx in &ready {
            remaining.remove(&idx);
            completed.insert(idx);
        }
        stages.push(ready);
    }

    let mut flow_to_stage = vec![0; num_flows];
    for (s_idx, stage_flows) in stages.iter().enumerate() {
        for &f_idx in stage_flows {
            flow_to_stage[f_idx] = s_idx;
        }
    }

    let mut downstream_flows: Vec<Vec<usize>> = vec![Vec::new(); num_flows];
    for (f_idx, deps) in all_flow_deps.iter().enumerate() {
        for &dep_f_idx in deps {
            downstream_flows[dep_f_idx].push(f_idx);
        }
    }

    // 1. Flow Execution Simulation Table
    let mut flow_table = Table::new()
        .box_style(&ROUNDED)
        .border_style(Style::parse("bright_cyan").unwrap_or_default())
        .header_style(Style::parse("bold white on blue").unwrap_or_default())
        .with_column(Column::new("Flow").width(6).justify(JustifyMethod::Center))
        .with_column(Column::new("Stage").width(14).justify(JustifyMethod::Center))
        .with_column(Column::new("Input Source").no_wrap())
        .with_column(Column::new("Execution Chain").no_wrap())
        .with_column(Column::new("Output Destination").no_wrap());

    for (f_idx, flow) in inspected_flows.iter().enumerate() {
        let chain_str = flow.steps_summary.join(" [cyan]>>[/] ");
        let s_idx = flow_to_stage.get(f_idx).copied().unwrap_or(0);
        let is_parallel = stages.get(s_idx).map(|s| s.len() > 1).unwrap_or(false);

        let stage_str = if is_parallel {
            format!("Stage {} [bold green](Par)[/]", s_idx + 1)
        } else {
            format!("Stage {}", s_idx + 1)
        };

        flow_table.add_row_markup([
            format!("#{}", flow.stmt_idx).as_str(),
            stage_str.as_str(),
            flow.source_desc.as_str(),
            chain_str.as_str(),
            flow.output_desc.as_str(),
        ]);
    }

    console.print_renderable(&flow_table);
    console.print("");

    // 2. Execution DAG & Concurrency Plan Tree
    let mut dag_tree = Tree::with_label(rich_rust::markup::render_or_plain(
        "[bold cyan]Pipeline Execution DAG[/]",
    ))
    .guides(TreeGuides::Rounded);

    for (s_idx, stage_flows) in stages.iter().enumerate() {
        let stage_num = s_idx + 1;
        let is_parallel = stage_flows.len() > 1;

        let stage_title = if is_parallel {
            format!("[bold yellow]Stage {}[/] [bold green](Parallel)[/]", stage_num)
        } else {
            format!("[bold yellow]Stage {}[/]", stage_num)
        };

        let mut stage_node = TreeNode::new(rich_rust::markup::render_or_plain(&stage_title));

        for &f_idx in stage_flows {
            let flow_num = f_idx + 1;
            let flow = &inspected_flows[f_idx];

            let flow_title = format!(
                "[bold white]Flow #{}[/]: [cyan]{}[/] [dim]→[/] [green]{}[/]",
                flow_num, flow.source_desc, flow.output_desc
            );
            let mut flow_node = TreeNode::new(rich_rust::markup::render_or_plain(&flow_title));

            // Trigger or Dependency
            let deps = &all_flow_deps[f_idx];
            if deps.is_empty() {
                let params_read: Vec<String> = all_flow_reads[f_idx]
                    .iter()
                    .filter(|v| ast.params.iter().any(|p| &p.name == *v))
                    .map(|v| format!("${}", v))
                    .collect();
                let trigger_desc = if params_read.is_empty() {
                    "[dim]Trigger:[/] [cyan]Pipeline Entry[/]".to_string()
                } else {
                    format!(
                        "[dim]Trigger:[/] [cyan]Pipeline Input[/] [dim]({})[/]",
                        params_read.join(", ")
                    )
                };
                flow_node = flow_node
                    .child(TreeNode::new(rich_rust::markup::render_or_plain(&trigger_desc)));
            } else {
                let mut dep_descriptions = Vec::new();
                let mut sorted_deps: Vec<usize> = deps.iter().copied().collect();
                sorted_deps.sort();
                for &dep_idx in &sorted_deps {
                    let vars = all_flow_dep_vars[f_idx]
                        .get(&dep_idx)
                        .map(|vs| {
                            vs.iter()
                                .map(|v| format!("${}", v))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();
                    dep_descriptions
                        .push(format!("[bold yellow]Flow #{}[/] [dim]({})[/]", dep_idx + 1, vars));
                }
                let dep_text = format!("[dim]Runs after:[/] {}", dep_descriptions.join(", "));
                flow_node =
                    flow_node.child(TreeNode::new(rich_rust::markup::render_or_plain(&dep_text)));
            }

            // Action chain
            let chain_str = flow.steps_summary.join(" [cyan]>>[/] ");
            let chain_text = format!("[dim]Actions:[/] [white]{}[/]", chain_str);
            flow_node =
                flow_node.child(TreeNode::new(rich_rust::markup::render_or_plain(&chain_text)));

            // Output
            let out_text = format!("[dim]Output:[/] [bold green]{}[/]", flow.output_desc);
            flow_node =
                flow_node.child(TreeNode::new(rich_rust::markup::render_or_plain(&out_text)));

            // Downstream triggers (if any)
            let downstream = &downstream_flows[f_idx];
            if !downstream.is_empty() {
                let mut sorted_downstream = downstream.clone();
                sorted_downstream.sort();
                let ds_str = sorted_downstream
                    .iter()
                    .map(|&ds_idx| {
                        let ds_stage = flow_to_stage[ds_idx] + 1;
                        format!(
                            "[bold yellow]Flow #{}[/] [dim](Stage {})[/]",
                            ds_idx + 1,
                            ds_stage
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let ds_text = format!("[dim]Triggers next:[/] {}", ds_str);
                flow_node = flow_node
                    .child(TreeNode::new(rich_rust::markup::render_or_plain(&ds_text)));
            }

            stage_node = stage_node.child(flow_node);
        }

        dag_tree = dag_tree.child(stage_node);
    }

    console.print_renderable(&dag_tree);
    console.print("");

    // 3. Action Dynamic Inspection Table (showing actual types from action binaries)
    if !action_names.is_empty() {
        console.print("[bold cyan]Native Action Binaries & Dynamic FFI Types:[/] \n");
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

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::DataType;

    #[test]
    fn test_datatype_bitmask_and_formatting() {
        let combined = DataType::RawBytes | DataType::Tensor;
        assert!(combined.contains(DataType::RawBytes));
        assert!(combined.contains(DataType::Tensor));
        assert!(!combined.contains(DataType::Audio));
        assert_eq!(format_data_type(combined), "RawBytes | Tensor");
        assert_eq!(format_data_type(DataType::Audio), "Audio");
        assert_eq!(format_data_type(DataType::Any), "Any");
    }

    #[test]
    fn test_are_types_compatible_audio_exclusivity() {
        let audio_action_input = DataType::Audio;
        let bridge_action_input = DataType::RawBytes | DataType::Tensor;
        let generic_tensor_input = DataType::Tensor;
        let to_tensor_input = DataType::Tensor | DataType::Image | DataType::Audio;

        // 1. Audio actions strictly accept Audio
        assert!(are_types_compatible(DataType::Audio, audio_action_input));
        assert!(!are_types_compatible(DataType::Tensor, audio_action_input), "Tensor must not be piped directly into Audio action");
        assert!(!are_types_compatible(DataType::RawBytes, audio_action_input), "RawBytes must not be piped directly into Audio action");
        assert!(!are_types_compatible(DataType::Image, audio_action_input), "Image must not be piped directly into Audio action");

        // 2. to_audio strictly accepts RawBytes and Tensor (Audio / Image cannot bypass)
        assert!(are_types_compatible(DataType::RawBytes, bridge_action_input));
        assert!(are_types_compatible(DataType::Tensor, bridge_action_input));
        assert!(!are_types_compatible(DataType::Audio, bridge_action_input));
        assert!(!are_types_compatible(DataType::Image, bridge_action_input));

        // 3. Tensor actions strictly accept Tensor (Audio and Image cannot be piped directly)
        assert!(!are_types_compatible(DataType::Audio, generic_tensor_input));
        assert!(!are_types_compatible(DataType::Image, generic_tensor_input));
        assert!(are_types_compatible(DataType::Tensor, generic_tensor_input));
        assert!(!are_types_compatible(DataType::RawBytes, generic_tensor_input));

        // 4. to_tensor explicitly accepts Tensor, Image, and Audio
        assert!(are_types_compatible(DataType::Audio, to_tensor_input));
        assert!(are_types_compatible(DataType::Image, to_tensor_input));
        assert!(are_types_compatible(DataType::Tensor, to_tensor_input));
        assert!(!are_types_compatible(DataType::RawBytes, to_tensor_input));

        // 5. Any accepts everything
        assert!(are_types_compatible(DataType::RawBytes, DataType::Any));
        assert!(are_types_compatible(DataType::Tensor, DataType::Any));
        assert!(are_types_compatible(DataType::Audio, DataType::Any));
        assert!(are_types_compatible(DataType::Image, DataType::Any));
    }
}
