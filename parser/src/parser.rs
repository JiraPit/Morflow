use chumsky::prelude::*;

use crate::ast::*;

fn ws<'a>() -> impl Parser<char, (), Error = Simple<char>> + Clone + 'a {
    let comment = choice((just('#').ignored(), just("//").ignored()))
        .then(take_until(text::newline()))
        .ignored();

    choice((text::whitespace().at_least(1).ignored(), comment))
        .repeated()
        .ignored()
}

fn padded<'a, T: 'a>(
    p: impl Parser<char, T, Error = Simple<char>> + Clone + 'a,
) -> impl Parser<char, T, Error = Simple<char>> + Clone + 'a {
    ws().ignore_then(p).then_ignore(ws())
}

pub fn parser() -> impl Parser<char, Pipeline, Error = Simple<char>> {
    // Identifiers (e.g. function names, parameter names)
    let ident = text::ident().map(|s: String| s);

    // Integers with optional + or -
    let signed_int = choice((just('+').to(1i64), just('-').to(-1i64), empty().to(1i64)))
        .then(text::int(10).map(|s: String| s.parse::<i64>().unwrap()))
        .map(|(sign, val)| sign * val);

    // Floats / Numbers
    let number = choice((
        just('+').to(1.0f64),
        just('-').to(-1.0f64),
        empty().to(1.0f64),
    ))
    .then(text::digits(10).then(just('.').then(text::digits(10))).map(
        |(int_part, (_, frac_part)): (String, (char, String))| {
            format!("{}.{}", int_part, frac_part)
                .parse::<f64>()
                .unwrap()
        },
    ))
    .map(|(sign, val)| Value::Float(sign * val))
    .or(signed_int.clone().map(Value::Int));

    // String literals
    let string_escape = just('\\').ignore_then(choice((
        just('\\'),
        just('/'),
        just('"'),
        just('\''),
        just('b').to('\x08'),
        just('f').to('\x0C'),
        just('n').to('\n'),
        just('r').to('\r'),
        just('t').to('\t'),
    )));

    let str_lit = just('"')
        .ignore_then(
            filter(|c| *c != '\\' && *c != '"')
                .or(string_escape)
                .repeated(),
        )
        .then_ignore(just('"'))
        .collect::<String>()
        .or(just('\'')
            .ignore_then(
                filter(|c| *c != '\\' && *c != '\'')
                    .or(string_escape)
                    .repeated(),
            )
            .then_ignore(just('\''))
            .collect::<String>())
        .map(Value::String);

    // Booleans
    let bool_lit = choice((
        text::keyword("true").to(Value::Bool(true)),
        text::keyword("false").to(Value::Bool(false)),
    ));

    // Slice item parser inside [...]
    let slice_named_dim = choice((
        text::keyword("dim").to("dim".to_string()),
        text::keyword("axis").to("axis".to_string()),
    ))
    .then_ignore(padded(just('=')))
    .then(signed_int.clone())
    .map(|(dim_name, index)| SliceItem::NamedDim { dim_name, index });

    let slice_full = just(':').map(|_| SliceItem::Full);

    let slice_range = signed_int
        .clone()
        .or_not()
        .then_ignore(just(':'))
        .then(signed_int.clone().or_not())
        .then(just(':').ignore_then(signed_int.clone()).or_not())
        .map(|((start, end), step)| SliceItem::Range { start, end, step });

    let slice_index = signed_int.map(SliceItem::Index);

    let slice_item = choice((slice_named_dim, slice_range, slice_full, slice_index));

    let slices = slice_item
        .separated_by(padded(just(',')))
        .delimited_by(just('['), just(']'));

    // Variable reference: $var_name or $var.field or $var[0:10]
    let var_ref = just('$')
        .ignore_then(ident)
        .then(just('.').ignore_then(ident).or_not())
        .then(slices.or_not())
        .map(|((name, field), slices_opt)| VarRef {
            name,
            field,
            slices: slices_opt.unwrap_or_default(),
        });

    let var_val = var_ref.clone().map(Value::Var);

    // Value
    let value = choice((number, str_lit, bool_lit, var_val));

    // Arguments in action call: foo(1, 2, bar=3)
    let named_arg = ident.then_ignore(padded(just('='))).then(value.clone());

    let positional_or_named_arg = choice((
        named_arg.map(|(k, v)| (Some(k), v)),
        value.clone().map(|v| (None, v)),
    ));

    let args_list = positional_or_named_arg
        .separated_by(padded(just(',')))
        .allow_trailing()
        .delimited_by(just('('), just(')'))
        .or_not()
        .map(|opt| {
            let mut pos = Vec::new();
            let mut named = Vec::new();
            if let Some(list) = opt {
                for (name_opt, val) in list {
                    match name_opt {
                        Some(k) => named.push((k, val)),
                        None => pos.push(val),
                    }
                }
            }
            (pos, named)
        });

    // Action call: e.g. load_audio("in.wav"), identity, or image_essentials.resize(512, 512)
    // Exclude reserved keywords: if, else, each, route, pipeline, accept, import, from, as, true, false
    let single_ident = ident.try_map(|name, span| {
        let is_reserved = matches!(
            name.as_str(),
            "if" | "else"
                | "each"
                | "route"
                | "pipeline"
                | "accept"
                | "import"
                | "from"
                | "as"
                | "true"
                | "false"
        );
        if is_reserved {
            Err(Simple::custom(
                span,
                format!("'{}' is a reserved keyword", name),
            ))
        } else {
            Ok(name)
        }
    });

    let action_ident = single_ident
        .then(just('.').ignore_then(single_ident).repeated())
        .map(|(first, rest)| {
            if rest.is_empty() {
                first
            } else {
                let mut full = first;
                for part in rest {
                    full.push('.');
                    full.push_str(&part);
                }
                full
            }
        });

    let action_call = action_ident
        .then(padded(args_list))
        .map(|(name, (pos, named))| ActionCall {
            name,
            positional_args: pos,
            named_args: named,
        });

    // Binary comparison operator
    let binary_op = choice((
        just("==").to(BinaryOp::Eq),
        just("!=").to(BinaryOp::NotEq),
        just("<=").to(BinaryOp::LtEq),
        just(">=").to(BinaryOp::GtEq),
        just('<').to(BinaryOp::Lt),
        just('>').to(BinaryOp::Gt),
    ));

    let condition = padded(value.clone())
        .then(padded(binary_op))
        .then(padded(value.clone()))
        .map(|((left, op), right)| Condition { left, op, right });

    // Recursive statement / flow parser
    let statement = recursive(|stmt| {
        let block = stmt
            .clone()
            .repeated()
            .delimited_by(padded(just('{')), padded(just('}')));

        // each ($var) { ... }
        let each_loop = text::keyword("each")
            .ignore_then(padded(
                just('$')
                    .ignore_then(ident)
                    .delimited_by(just('('), just(')')),
            ))
            .then(block.clone())
            .map(|(var_name, body)| FlowStep::Each(EachLoop { var_name, body }));

        // if (cond) { ... } else { ... }
        let if_else = text::keyword("if")
            .ignore_then(padded(condition.clone().delimited_by(just('('), just(')'))))
            .then(block.clone())
            .then(
                padded(text::keyword("else"))
                    .ignore_then(block.clone())
                    .or_not(),
            )
            .map(|((cond, then_b), else_b)| {
                FlowStep::IfElse(IfElseBranch {
                    condition: cond,
                    then_branch: then_b,
                    else_branch: else_b,
                })
            });

        // route { cond => { ... } ... else => { ... } }
        let route_arm = condition
            .then_ignore(padded(just("=>")))
            .then(block.clone().or(stmt.clone().map(|s| vec![s])))
            .map(|(cond, body)| RouteArm {
                condition: cond,
                body,
            });

        let default_arm = text::keyword("else")
            .then_ignore(padded(just("=>")))
            .then(block.clone().or(stmt.clone().map(|s| vec![s])))
            .map(|(_, body)| body);

        let route_block = text::keyword("route")
            .ignore_then(padded(just('{')))
            .ignore_then(route_arm.repeated())
            .then(default_arm.or_not())
            .then_ignore(padded(just('}')))
            .map(|(arms, default_arm)| FlowStep::Route(RouteBranch { arms, default_arm }));

        // Mid-stream tap: $var_name
        let tap_step = just('$').ignore_then(ident).map(FlowStep::Tap);

        // Var step at the start of a chain: $raw_audio[0:100]
        let var_step = var_ref.clone().map(FlowStep::Var);

        let action_step = action_call.clone().map(FlowStep::Action);

        let flow_step = choice((
            each_loop.clone(),
            if_else.clone(),
            route_block.clone(),
            var_step.clone(),
            action_step.clone(),
        ));

        // Flow step following '>>' can also be a tap $var
        let next_step = choice((
            each_loop,
            if_else,
            route_block,
            tap_step,
            var_step,
            action_step,
        ));

        let flow_chain = flow_step
            .then(padded(just(">>")).ignore_then(padded(next_step)).repeated())
            .map(|(first, mut rest)| {
                let mut steps = vec![first];
                steps.append(&mut rest);
                Statement::Flow(FlowChain { steps })
            });

        padded(flow_chain)
    });

    // Pipeline parameter parsing
    let param_name = choice((just('$').ignore_then(ident), ident));

    // accept $a or accept $b = 44100
    let accept_stmt = text::keyword("accept")
        .ignore_then(padded(param_name))
        .then(padded(just('=')).ignore_then(value.clone()).or_not())
        .map(|(name, default_value)| PipelineParam {
            name,
            default_value,
        });

    // Version string parser: e.g. "latest", "v1", "1.0.0"
    let version_part = text::digits(10).or(ident);
    let version_str = version_part
        .separated_by(just('.'))
        .at_least(1)
        .map(|parts| parts.join("."));

    // Import item: color_adjust [as ca]
    let import_item = ident
        .then(padded(text::keyword("as")).ignore_then(padded(ident)).or_not())
        .map(|(name, alias)| ImportItem { name, alias });

    let import_items_list = import_item
        .separated_by(padded(just(',')))
        .allow_trailing();

    // from <package>.<version> import <item1>, <item2>
    let from_import = text::keyword("from")
        .ignore_then(padded(ident))
        .then_ignore(just('.'))
        .then(padded(version_str.clone()))
        .then_ignore(padded(text::keyword("import")))
        .then(padded(import_items_list))
        .map(|((package, version), items)| {
            ImportStmt::Items(ItemsImport {
                package,
                version,
                items,
            })
        });

    // import <package>.<version> [as <alias>]
    let pkg_import = text::keyword("import")
        .ignore_then(padded(ident))
        .then_ignore(just('.'))
        .then(padded(version_str))
        .then(padded(text::keyword("as")).ignore_then(padded(ident)).or_not())
        .map(|((package, version), alias)| {
            ImportStmt::Package(PackageImport {
                package,
                version,
                alias,
            })
        });

    let import_stmt = choice((from_import, pkg_import));

    enum TopLevel {
        Import(ImportStmt),
        Param(PipelineParam),
        Stmt(Statement),
    }

    let top_level_item = choice((
        padded(import_stmt).map(TopLevel::Import),
        padded(accept_stmt).map(TopLevel::Param),
        statement.clone().map(TopLevel::Stmt),
    ));

    // Top-level pipeline format with imports, `accept $a` declarations and flows
    let pipeline_top_level = top_level_item.repeated().map(|items| {
        let mut imports = Vec::new();
        let mut params = Vec::new();
        let mut statements = Vec::new();
        for item in items {
            match item {
                TopLevel::Import(imp) => imports.push(imp),
                TopLevel::Param(param) => params.push(param),
                TopLevel::Stmt(stmt) => statements.push(stmt),
            }
        }
        Pipeline {
            imports,
            params,
            statements,
        }
    });

    padded(pipeline_top_level).then_ignore(end())
}
