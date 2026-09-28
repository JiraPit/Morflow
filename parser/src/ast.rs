use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pipeline {
    pub imports: Vec<ImportStmt>,
    pub params: Vec<PipelineParam>,
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportStmt {
    /// `import <package>.<version> [as <alias>]`
    Package(PackageImport),
    /// `from <package>.<version> import <item1>, <item2>`
    Items(ItemsImport),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackageImport {
    pub package: String,
    pub version: String,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemsImport {
    pub package: String,
    pub version: String,
    pub items: Vec<ImportItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportItem {
    pub name: String,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineParam {
    pub name: String,
    pub default_value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Flow(FlowChain),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowChain {
    pub steps: Vec<FlowStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FlowStep {
    /// An action call: `load_audio("sample.wav")` or `resample(rate=44100)` or `identity`
    Action(ActionCall),
    /// Mid-stream tap: `>> $var_name`
    Tap(String),
    /// Variable reference (starting a stream): `$raw_audio` or `$raw_audio[0:100]`
    Var(VarRef),
    /// Explicitly named each loop: `each ($var) { ... }`
    Each(EachLoop),
    /// Conditional branch: `if (cond) { ... } else { ... }`
    IfElse(IfElseBranch),
    /// Multi-route matching: `route { ... }`
    Route(RouteBranch),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionCall {
    pub name: String,
    pub positional_args: Vec<Value>,
    pub named_args: Vec<(String, Value)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarRef {
    pub name: String,
    pub field: Option<String>,
    pub slices: Vec<SliceItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SliceItem {
    /// Exact integer index, e.g. `2` or `-1`
    Index(i64),
    /// Range slice, e.g. `0:48000`, `:100`, `50:`, `::2`
    Range {
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
    },
    /// Full dimension slice `:`
    Full,
    /// Named dimension index, e.g. `dim=2` or `axis=0`
    NamedDim { dim_name: String, index: i64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EachLoop {
    pub var_name: String,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfElseBranch {
    pub condition: Condition,
    pub then_branch: Vec<Statement>,
    pub else_branch: Option<Vec<Statement>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteBranch {
    pub arms: Vec<RouteArm>,
    pub default_arm: Option<Vec<Statement>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteArm {
    pub condition: Condition,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    pub left: Value,
    pub op: BinaryOp,
    pub right: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Eq,    // ==
    NotEq, // !=
    Lt,    // <
    LtEq,  // <=
    Gt,    // >
    GtEq,  // >=
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Var(VarRef),
}
