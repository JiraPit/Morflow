use pipeline::{check::ShapeCoverage, ActionIdentity, ActionRegistry};
use std::collections::HashMap;

fn coverage(source: &str) -> ShapeCoverage {
    let ast = parser::parse(source).unwrap();
    static REGISTRY: std::sync::OnceLock<ActionRegistry> = std::sync::OnceLock::new();
    let registry = REGISTRY.get_or_init(ActionRegistry::default);
    let mut loaded = HashMap::new();
    for (pack, name) in [
        ("base", "identity"),
        ("tensor_essentials", "reshape"),
        ("linalg_essentials", "matmul"),
    ] {
        loaded.insert(
            name.to_string(),
            registry
                .get_or_load(&ActionIdentity::new(pack, "latest", name).unwrap())
                .unwrap(),
        );
    }
    ShapeCoverage::analyze(&ast, &loaded).unwrap()
}
#[test]
fn concrete_dynamic_and_invalid_calls_have_distinct_coverage() {
    let report = coverage(
        r#"
accept Tensor[2,3] $fixed
accept Tensor[*,3] $dynamic
$fixed >> reshape("3,2") >> emit
$dynamic >> reshape("3,2") >> emit
$fixed >> reshape("4,2") >> emit
"#,
    );
    assert_eq!(report.actions.len(), 3);
    assert_eq!(report.ready(), 1);
    assert_eq!(report.percentage(), Some(100.0 / 3.0));
    assert!(report.has_invalid());
    assert_eq!(
        report.actions.iter().map(|a| a.status).collect::<Vec<_>>(),
        ["Ready", "Deferred", "Invalid"]
    );
}
#[test]
fn repeated_calls_are_counted_not_unique_action_names_or_loop_iterations() {
    let report = coverage(
        r#"
accept Tensor[100,3] $x
$x >> each ($row) {
  $row >> identity >> each ($item) { $item >> identity }
} >> identity >> emit
"#,
    );
    assert_eq!(report.actions.len(), 3);
    assert_eq!(report.percentage(), Some(100.0));
    assert!(report.actions[1].location.matches("/each").count() == 2);
}
#[test]
fn every_conditional_and_route_arm_is_included_and_joined_conservatively() {
    let report = coverage(
        r#"
accept Tensor[2,3] $x
accept IntArg $choice
$x >> if ($choice == 0) {
  reshape("3,2")
} else {
  reshape("1,6")
} >> identity
$x >> route {
  $choice == 0 => { identity }
  $choice == 1 => { reshape("6") }
  else => { identity }
} >> identity
"#,
    );
    assert_eq!(report.actions.len(), 7);
    assert_eq!(report.ready(), 5);
    assert_eq!(report.actions[2].status, "Deferred");
    assert_eq!(report.actions[6].status, "Deferred");
    assert!(report.actions[0].location.contains("then"));
    assert!(report.actions[4].location.contains("arm 2"));
}
#[test]
fn dynamic_host_arguments_remain_dynamic_even_with_defaults() {
    let report = coverage(
        r#"
accept Tensor[2,3] $x
accept StrArg $target = "3,2"
$x >> reshape($target) >> identity >> emit
"#,
    );
    assert_eq!(report.percentage(), Some(0.0));
    assert!(report.actions.iter().all(|a| a.status == "Deferred"));
}
#[test]
fn ordered_composite_inputs_and_indexing_retain_shape_guarantees() {
    let report = coverage(
        r#"
accept Composite[Tensor[2,3], Tensor[3,4]] $parts
$parts >> matmul >> emit
$parts[0] >> identity >> emit
"#,
    );
    assert_eq!(report.percentage(), Some(100.0));
    assert_eq!(
        report.actions[0].output.to_ptype().to_string(),
        "Tensor[2, 4]"
    );
}
#[test]
fn builtins_only_are_not_vacuously_one_hundred_percent() {
    let report = coverage("accept Tensor[2,3] $x\n$x >> emit\n");
    assert!(report.actions.is_empty());
    assert_eq!(report.percentage(), None);
}

#[test]
fn cli_uses_the_recursive_loop_output_shape_for_following_actions() {
    let source = r#"
from base/latest import identity
from tensor_essentials/latest import reshape
accept Tensor[2,6] $x
accept IntArg $choice
$x >> each ($row) {
  if ($choice == 0) { $row[0:3] >> identity }
  else { $row[0:3] >> identity }
} >> reshape("6") >> emit
"#;
    let report = coverage(source);
    assert_eq!(report.percentage(), Some(100.0));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("loop.morf");
    std::fs::write(&path, source).unwrap();
    pipeline::check::check_pipeline(&path, None).unwrap();
}
