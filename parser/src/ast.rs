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
    pub param_type: ParamType,
    pub default_value: Option<Value>,
}

/// A dimension of a declared shape: a wildcard or an exact length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamDim {
    /// `*` — any length.
    Any,
    /// An exact length, as in the `3` of `Tensor[*,*,3]`.
    Fixed(usize),
}

/// A shape as declared on a parameter: an unknown rank, or a fixed rank whose
/// dimensions may contain wildcards.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParamShape {
    /// Rank is not pinned: `Tensor`, `Image`, `Audio`.
    #[default]
    AnyRank,
    /// Rank is pinned: `Tensor[2,3]`, `Image[*,*,3]`, `Tensor[rank=2]`.
    Ranked { dims: Vec<ParamDim> },
}

/// The declared type of a pipeline parameter.
///
/// Every `accept` names one of these; there is no untyped parameter, because
/// the type is what lets the checker reason about what flows where.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamType {
    /// An opaque byte buffer, e.g. a WAV file's contents.
    Bytes,
    /// An integer argument. Arguments are readable by actions but never flow
    /// through a chain.
    IntArg,
    /// A floating-point argument.
    FloatArg,
    /// A string argument.
    StrArg,
    /// A boolean argument.
    BoolArg,
    /// A rank-0 numeric value.
    Scalar,
    /// A tensor, with an optional shape specification.
    Tensor(ParamShape),
    /// An image, with an optional shape specification. The channel dimension is
    /// validated against the payload's own layout.
    Image(ParamShape),
    /// An audio payload, with an optional shape specification. Channel and
    /// sample dimensions are validated against the payload's own layout.
    Audio(ParamShape),
    /// A tuple of payloads.
    Composite,
    /// Ordered component declarations, optionally nested.
    CompositeItems(Vec<ParamType>),
}

impl ParamType {
    /// The type's keyword, as written in a declaration.
    pub fn keyword(&self) -> &'static str {
        match self {
            ParamType::Bytes => "Bytes",
            ParamType::IntArg => "IntArg",
            ParamType::FloatArg => "FloatArg",
            ParamType::StrArg => "StrArg",
            ParamType::BoolArg => "BoolArg",
            ParamType::Scalar => "Scalar",
            ParamType::Tensor(_) => "Tensor",
            ParamType::Image(_) => "Image",
            ParamType::Audio(_) => "Audio",
            ParamType::Composite | ParamType::CompositeItems(_) => "Composite",
        }
    }

    /// Whether this is one of the argument types, which are never flowable.
    pub fn is_arg(&self) -> bool {
        matches!(
            self,
            ParamType::IntArg | ParamType::FloatArg | ParamType::StrArg | ParamType::BoolArg
        )
    }
}

impl std::fmt::Display for ParamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = match self {
            ParamType::Tensor(shape) | ParamType::Image(shape) | ParamType::Audio(shape) => shape,
            _ => return write!(f, "{}", self.keyword()),
        };
        match shape {
            ParamShape::AnyRank => write!(f, "{}", self.keyword()),
            // All-wildcard ranks are written back in their short form.
            ParamShape::Ranked { dims } if dims.iter().all(|d| *d == ParamDim::Any) => {
                write!(f, "{}[rank={}]", self.keyword(), dims.len())
            }
            ParamShape::Ranked { dims } => {
                let parts: Vec<String> = dims
                    .iter()
                    .map(|d| match d {
                        ParamDim::Any => "*".to_string(),
                        ParamDim::Fixed(n) => n.to_string(),
                    })
                    .collect();
                write!(f, "{}[{}]", self.keyword(), parts.join(", "))
            }
        }
    }
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
    /// One slice group per pair of brackets; each group restarts at axis 0.
    pub slices: Vec<Vec<SliceItem>>,
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
