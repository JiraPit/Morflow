//! Tests call the exported functions exactly as a C caller would. Tests that
//! need action binaries skip unless `MORFLOW_ACTIONS_PATH` is set.

use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Building inputs

/// Owns everything a [`MorflowValueIn`] points at, so the raw values stay
/// valid while a test uses them.
#[derive(Default)]
struct Arena {
    strings: Vec<CString>,
    bytes: Vec<Vec<u8>>,
    shapes: Vec<Vec<u64>>,
    lists: Vec<Vec<MorflowValueIn>>,
}

#[derive(Default, Clone)]
struct Spec {
    kind: u32,
    data: Vec<u8>,
    shape: Vec<u64>,
    dtype: Option<&'static str>,
    payload_type: Option<&'static str>,
    color_space: Option<&'static str>,
    layout: Option<&'static str>,
    sample_rate: Option<i64>,
    channels: Option<i64>,
    items: Vec<Spec>,
    number: f64,
    flag: bool,
}

impl Spec {
    fn tensor(data: Vec<u8>, shape: &[u64], dtype: &'static str) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_TENSOR,
            data,
            shape: shape.to_vec(),
            dtype: Some(dtype),
            ..Spec::default()
        }
    }

    fn f32s(values: &[f32], shape: &[u64]) -> Spec {
        Spec::tensor(
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            shape,
            "f32",
        )
    }

    fn composite(items: Vec<Spec>) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_TENSOR,
            payload_type: Some("composite"),
            items,
            ..Spec::default()
        }
    }

    fn bytes(data: &[u8]) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_BYTES,
            data: data.to_vec(),
            ..Spec::default()
        }
    }

    fn number(value: f64) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_NUMBER,
            number: value,
            ..Spec::default()
        }
    }

    fn text(value: &str) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_TEXT,
            data: value.as_bytes().to_vec(),
            ..Spec::default()
        }
    }

    fn boolean(value: bool) -> Spec {
        Spec {
            kind: MORFLOW_VALUE_BOOL,
            flag: value,
            ..Spec::default()
        }
    }

    fn with(mut self, f: impl FnOnce(&mut Spec)) -> Spec {
        f(&mut self);
        self
    }
}

impl Arena {
    fn str(&mut self, text: Option<&str>) -> *const c_char {
        match text {
            None => ptr::null(),
            Some(text) => {
                self.strings.push(CString::new(text).unwrap());
                self.strings.last().unwrap().as_ptr()
            }
        }
    }

    fn value(&mut self, spec: &Spec) -> MorflowValueIn {
        let items: Vec<MorflowValueIn> = spec.items.iter().map(|item| self.value(item)).collect();
        self.lists.push(items);
        let items = self.lists.last().unwrap();
        let (items_ptr, item_count) = (items.as_ptr(), items.len());
        self.bytes.push(spec.data.clone());
        let data = self.bytes.last().unwrap();
        let (data_ptr, data_len) = (data.as_ptr(), data.len());
        self.shapes.push(spec.shape.clone());
        let shape = self.shapes.last().unwrap();
        let (shape_ptr, rank) = (shape.as_ptr(), shape.len());
        MorflowValueIn {
            kind: spec.kind,
            data: data_ptr,
            data_len,
            shape: shape_ptr,
            rank,
            dtype: self.str(spec.dtype),
            payload_type: self.str(spec.payload_type),
            color_space: self.str(spec.color_space),
            layout: self.str(spec.layout),
            sample_rate: spec.sample_rate.unwrap_or(-1),
            channels: spec.channels.unwrap_or(-1),
            items: items_ptr,
            item_count,
            number: spec.number,
            flag: spec.flag as u8,
        }
    }

    fn list(&mut self, specs: &[Spec]) -> Vec<MorflowValueIn> {
        specs.iter().map(|spec| self.value(spec)).collect()
    }
}

// ---------------------------------------------------------------------------
// Reading results

#[derive(Debug, PartialEq)]
enum Out {
    Tensor {
        dtype: String,
        shape: Vec<u64>,
        data: Vec<u8>,
    },
    Composite(Vec<Out>),
}

impl Out {
    fn f32s(&self) -> Vec<f32> {
        match self {
            Out::Tensor { dtype, data, .. } => {
                assert_eq!(dtype, "f32");
                let (chunks, rest) = data.as_chunks::<4>();
                assert!(rest.is_empty());
                chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect()
            }
            other => panic!("expected a tensor, got {:?}", other),
        }
    }

    fn shape(&self) -> &[u64] {
        match self {
            Out::Tensor { shape, .. } => shape,
            other => panic!("expected a tensor, got {:?}", other),
        }
    }

    fn data(&self) -> &[u8] {
        match self {
            Out::Tensor { data, .. } => data,
            other => panic!("expected a tensor, got {:?}", other),
        }
    }
}

unsafe fn read_value(value: &MorflowValueOut) -> Out {
    let dtype = CStr::from_ptr(value.dtype).to_str().unwrap().to_string();
    let shape = slice(value.shape, value.rank).to_vec();
    if dtype == "composite" {
        assert_eq!(value.data_len, 0);
        assert_eq!(shape, vec![value.item_count as u64]);
        return Out::Composite(
            slice(value.items, value.item_count)
                .iter()
                .map(|v| read_value(v))
                .collect(),
        );
    }
    Out::Tensor {
        dtype,
        shape,
        data: slice(value.data, value.data_len).to_vec(),
    }
}

/// Reads and frees a result.
unsafe fn take(result: *mut MorflowResult) -> Result<Vec<(Option<String>, Out)>, String> {
    assert!(!result.is_null());
    let error = morflow_result_error(result);
    let outcome = if !error.is_null() {
        assert_eq!(morflow_result_count(result), 0);
        Err(CStr::from_ptr(error).to_str().unwrap().to_string())
    } else {
        let count = morflow_result_count(result);
        assert!(morflow_result_value(result, count).is_null());
        assert!(morflow_result_name(result, count).is_null());
        Ok((0..count)
            .map(|i| {
                let name = morflow_result_name(result, i);
                let name =
                    (!name.is_null()).then(|| CStr::from_ptr(name).to_str().unwrap().to_string());
                (name, read_value(&*morflow_result_value(result, i)))
            })
            .collect())
    };
    morflow_result_free(result);
    outcome
}

unsafe fn single(result: *mut MorflowResult) -> Result<Out, String> {
    let mut values = take(result)?;
    assert_eq!(values.len(), 1);
    let (name, value) = values.remove(0);
    assert_eq!(name, None);
    Ok(value)
}

unsafe fn named(result: *mut MorflowResult) -> Result<HashMap<String, Out>, String> {
    let values = take(result)?;
    let names: Vec<_> = values.iter().map(|(n, _)| n.clone().unwrap()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "all-outputs results are sorted by name");
    Ok(values.into_iter().map(|(n, v)| (n.unwrap(), v)).collect())
}

// ---------------------------------------------------------------------------
// Pipelines

struct Pipeline(*mut MorflowPipelineHandle);

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { morflow_pipeline_free(self.0) }
    }
}

// The handle is used from worker threads in the concurrency tests.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

fn compile(source: &str) -> Result<Pipeline, String> {
    let source = CString::new(source).unwrap();
    let mut error: *mut c_char = ptr::null_mut();
    let handle = unsafe { morflow_from_str(source.as_ptr(), &mut error) };
    if handle.is_null() {
        assert!(!error.is_null());
        let message = unsafe { CStr::from_ptr(error) }
            .to_str()
            .unwrap()
            .to_string();
        unsafe { morflow_string_free(error) };
        Err(message)
    } else {
        assert!(error.is_null());
        Ok(Pipeline(handle))
    }
}

fn pipeline(source: &str) -> Pipeline {
    compile(source).unwrap()
}

impl Pipeline {
    fn run(&self, input: Option<Spec>) -> Result<Out, String> {
        let mut arena = Arena::default();
        let value = input.map(|spec| arena.value(&spec));
        let ptr = value.as_ref().map_or(ptr::null(), |v| v as *const _);
        unsafe { single(morflow_run(self.0, ptr, 0)) }
    }

    fn run_all(&self, input: Spec) -> Result<HashMap<String, Out>, String> {
        let mut arena = Arena::default();
        let value = arena.value(&input);
        unsafe { named(morflow_run(self.0, &value, 1)) }
    }

    fn run_args(&self, args: &[Spec]) -> Result<Out, String> {
        let mut arena = Arena::default();
        let values = arena.list(args);
        unsafe { single(morflow_run_args(self.0, values.as_ptr(), values.len(), 0)) }
    }

    fn params(&self) -> Vec<String> {
        let count = unsafe { morflow_param_count(self.0) };
        assert!(unsafe { morflow_param_name(self.0, count) }.is_null());
        (0..count)
            .map(|i| unsafe {
                let name = morflow_param_name(self.0, i);
                let text = CStr::from_ptr(name).to_str().unwrap().to_string();
                morflow_string_free(name);
                text
            })
            .collect()
    }
}

/// Actions come from a prepared action cache or from bundled pack libraries.
fn actions_available() -> bool {
    let available = std::env::var_os("MORFLOW_ACTIONS_PATH").is_some()
        || std::env::var_os("MORFLOW_PACKS_PATH").is_some();
    if !available {
        eprintln!("skipped: set MORFLOW_ACTIONS_PATH to a prepared action cache, or MORFLOW_PACKS_PATH to pack libraries");
    }
    available
}

// ---------------------------------------------------------------------------
// Async delivery

fn deliveries() -> &'static (Mutex<HashMap<i64, usize>>, Condvar) {
    static DELIVERIES: OnceLock<(Mutex<HashMap<i64, usize>>, Condvar)> = OnceLock::new();
    DELIVERIES.get_or_init(|| (Mutex::new(HashMap::new()), Condvar::new()))
}

extern "C" fn deliver(request_id: i64, result: *mut MorflowResult) {
    let (lock, ready) = deliveries();
    let previous = lock.lock().unwrap().insert(request_id, result as usize);
    assert!(
        previous.is_none(),
        "callback fired twice for request {}",
        request_id
    );
    ready.notify_all();
}

fn next_request() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn wait_for(request_id: i64) -> *mut MorflowResult {
    let (lock, ready) = deliveries();
    let (mut delivered, timeout) = ready
        .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(30), |d| {
            !d.contains_key(&request_id)
        })
        .unwrap();
    assert!(
        !timeout.timed_out(),
        "request {} never completed",
        request_id
    );
    delivered.remove(&request_id).unwrap() as *mut MorflowResult
}

fn was_delivered(request_id: i64) -> bool {
    deliveries().0.lock().unwrap().contains_key(&request_id)
}

const ECHO: &str = "accept Tensor $data\n$data >> emit";
const ECHO_COMPOSITE: &str = "accept Composite $pair\n$pair >> emit";
const ECHO_BYTES: &str = "accept Bytes $data\n$data >> emit";

// ---------------------------------------------------------------------------
// Compiling and metadata

#[test]
fn version_matches_the_crate() {
    let version = unsafe { CStr::from_ptr(morflow_version()) };
    assert_eq!(version.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn version_matches_the_engine() {
    // The Dart package checks the library's version, so the C interface must
    // be bumped together with the engine.
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../pipeline/Cargo.toml"
    ))
    .unwrap();
    let engine = manifest
        .lines()
        .find_map(|line| line.strip_prefix("version = \""))
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap();
    assert_eq!(env!("CARGO_PKG_VERSION"), engine);
}

#[test]
fn parameters_are_listed_in_declaration_order() {
    let p = pipeline("accept Audio $audio_in\naccept IntArg $rate = 44100\n$audio_in >> emit");
    assert_eq!(p.params(), vec!["audio_in", "rate"]);
}

#[test]
fn invalid_syntax_reports_a_parse_error() {
    let error = compile("accept Tensor $x >>> broken").err().unwrap();
    assert!(error.starts_with("Parse error"), "{}", error);
}

#[test]
fn undeclared_actions_report_their_imports() {
    let error = compile("accept Tensor $data\n$data >> identity >> emit")
        .err()
        .unwrap();
    assert!(error.contains("not declared by the imports"), "{}", error);
}

#[test]
fn loading_a_missing_file_fails() {
    let path = CString::new("/definitely/missing/pipeline.morf").unwrap();
    let mut error: *mut c_char = ptr::null_mut();
    let handle = unsafe { morflow_load(path.as_ptr(), &mut error) };
    assert!(handle.is_null());
    let message = unsafe { CStr::from_ptr(error) }
        .to_str()
        .unwrap()
        .to_string();
    unsafe { morflow_string_free(error) };
    assert!(message.starts_with("IO error"), "{}", message);
}

#[test]
fn loading_a_file_compiles_it() {
    let dir = std::env::temp_dir().join(format!("morflow-capi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("echo.morf");
    std::fs::write(&file, ECHO).unwrap();
    let path = CString::new(file.to_str().unwrap()).unwrap();
    let handle = unsafe { morflow_load(path.as_ptr(), ptr::null_mut()) };
    assert!(!handle.is_null());
    let p = Pipeline(handle);
    assert_eq!(p.params(), vec!["data"]);
    assert_eq!(
        p.run(Some(Spec::f32s(&[1.0, 2.0], &[2]))).unwrap().f32s(),
        vec![1.0, 2.0]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn null_arguments_are_errors_not_crashes() {
    unsafe {
        let mut error: *mut c_char = ptr::null_mut();
        assert!(morflow_from_str(ptr::null(), &mut error).is_null());
        assert_eq!(
            CStr::from_ptr(error).to_str().unwrap(),
            "Pipeline source is null"
        );
        morflow_string_free(error);
        // A null error slot is allowed.
        assert!(morflow_load(ptr::null(), ptr::null_mut()).is_null());

        let error = single(morflow_run(ptr::null(), ptr::null(), 0))
            .err()
            .unwrap();
        assert_eq!(error, "Pipeline handle is null");
        assert_eq!(morflow_param_count(ptr::null()), 0);
        assert!(morflow_param_name(ptr::null(), 0).is_null());
        morflow_pipeline_free(ptr::null_mut());
        morflow_result_free(ptr::null_mut());
        morflow_string_free(ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// Values in and out

#[test]
fn every_dtype_round_trips() {
    let p = pipeline(ECHO);
    let cases: Vec<(&'static str, Vec<u8>)> = vec![
        (
            "f32",
            [1.5f32, -2.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (
            "f64",
            [1.5f64, -2.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        ("u8", vec![7, 8]),
        (
            "i8",
            [-3i8, 4].iter().flat_map(|v| v.to_ne_bytes()).collect(),
        ),
        (
            "i16",
            [-300i16, 400]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (
            "i32",
            [-70000i32, 80000]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (
            "i64",
            [-(1i64 << 40), 1 << 40]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (
            "u32",
            [1u32, 4_000_000_000]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (
            "u64",
            [1u64, 1 << 63]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
    ];
    for (dtype, data) in cases {
        let out = p
            .run(Some(Spec::tensor(data.clone(), &[2], dtype)))
            .unwrap();
        assert_eq!(
            out,
            Out::Tensor {
                dtype: dtype.to_string(),
                shape: vec![2],
                data
            }
        );
    }
}

#[test]
fn dtype_defaults_to_f32() {
    let out = pipeline(ECHO)
        .run(Some(Spec::f32s(&[3.0], &[1]).with(|s| s.dtype = None)))
        .unwrap();
    assert_eq!(out.f32s(), vec![3.0]);
}

#[test]
fn multi_dimensional_slices_keep_their_shape() {
    let p = pipeline("accept Tensor $img\n$img[1:3, :] >> emit");
    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = p.run(Some(Spec::f32s(&values, &[4, 4]))).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    assert_eq!(out.f32s(), (4..12).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn composites_round_trip_in_order() {
    let first = Spec::tensor(
        [1.5f64, 2.5].iter().flat_map(|v| v.to_ne_bytes()).collect(),
        &[2],
        "f64",
    );
    let second = Spec::tensor(7i64.to_ne_bytes().to_vec(), &[1], "i64");
    let out = pipeline(ECHO_COMPOSITE)
        .run(Some(Spec::composite(vec![first, second])))
        .unwrap();
    match out {
        Out::Composite(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(
                items[0],
                Out::Tensor {
                    dtype: "f64".into(),
                    shape: vec![2],
                    data: [1.5f64, 2.5].iter().flat_map(|v| v.to_ne_bytes()).collect()
                }
            );
            assert_eq!(
                items[1],
                Out::Tensor {
                    dtype: "i64".into(),
                    shape: vec![1],
                    data: 7i64.to_ne_bytes().to_vec()
                }
            );
        }
        other => panic!("expected a composite, got {:?}", other),
    }
}

#[test]
fn nested_composites_round_trip() {
    let inner = Spec::composite(vec![Spec::f32s(&[1.0], &[1])]);
    let out = pipeline(ECHO_COMPOSITE)
        .run(Some(Spec::composite(vec![inner, Spec::f32s(&[2.0], &[1])])))
        .unwrap();
    match out {
        Out::Composite(items) => {
            assert!(matches!(&items[0], Out::Composite(inner) if inner[0].f32s() == vec![1.0]));
            assert_eq!(items[1].f32s(), vec![2.0]);
        }
        other => panic!("expected a composite, got {:?}", other),
    }
}

#[test]
fn bytes_arrive_as_a_u8_tensor() {
    let out = pipeline(ECHO_BYTES)
        .run(Some(Spec::bytes(&[1, 2, 3])))
        .unwrap();
    assert_eq!(
        out,
        Out::Tensor {
            dtype: "u8".into(),
            shape: vec![3],
            data: vec![1, 2, 3]
        }
    );
}

#[test]
fn no_input_is_an_empty_byte_buffer() {
    let out = pipeline(ECHO_BYTES).run(None).unwrap();
    assert_eq!(
        out,
        Out::Tensor {
            dtype: "u8".into(),
            shape: vec![0],
            data: vec![]
        }
    );
}

#[test]
fn numbers_strings_and_booleans_bind_as_text_like_nodejs() {
    let p = pipeline(ECHO_BYTES);
    for (spec, text) in [
        (Spec::number(3.0), "3"),
        (Spec::number(-3.0), "-3"),
        (Spec::number(2.5), "2.5"),
        (Spec::number(44100.0), "44100"),
        (Spec::text("hann"), "hann"),
        (Spec::text("ünïcödé"), "ünïcödé"),
        (Spec::boolean(true), "true"),
        (Spec::boolean(false), "false"),
    ] {
        let out = p.run_args(&[spec]).unwrap();
        assert_eq!(std::str::from_utf8(out.data()).unwrap(), text);
    }
}

#[test]
fn named_outputs_are_returned_sorted() {
    let p = pipeline(
        "accept Tensor $audio\n$audio[0:2] >> emit(\"low\")\n$audio[2:4] >> emit(\"high\")",
    );
    let outs = p.run_all(Spec::f32s(&[1.0, 2.0, 3.0, 4.0], &[4])).unwrap();
    assert_eq!(outs.len(), 2);
    assert_eq!(outs["low"].f32s(), vec![1.0, 2.0]);
    assert_eq!(outs["high"].f32s(), vec![3.0, 4.0]);
}

#[test]
fn single_output_runs_reject_several_outputs() {
    let p = pipeline("accept Tensor $t\n$t[0:1] >> emit(\"a\")\n$t[1:2] >> emit(\"b\")");
    let error = p.run(Some(Spec::f32s(&[1.0, 2.0], &[2]))).err().unwrap();
    assert!(
        error.contains("Expected a single emitted output"),
        "{}",
        error
    );
}

// ---------------------------------------------------------------------------
// Input validation (same messages as the Node.js binding)

fn run_error(input: Spec) -> String {
    pipeline(ECHO).run(Some(input)).err().unwrap()
}

#[test]
fn unknown_dtype_is_rejected() {
    let error = run_error(Spec::tensor(vec![0; 8], &[2], "f16"));
    assert!(error.starts_with("Unknown dtype 'f16'"), "{}", error);
}

#[test]
fn data_must_fill_whole_elements() {
    let error = run_error(Spec::tensor(vec![0; 6], &[2], "f32"));
    assert_eq!(
        error,
        "Data buffer size is not a multiple of 4 for a f32 tensor"
    );
}

#[test]
fn shape_must_match_the_data() {
    let error = run_error(Spec::f32s(&[1.0, 2.0], &[3]));
    assert!(!error.is_empty());
}

#[test]
fn unknown_payload_type_is_rejected() {
    let error = run_error(Spec::f32s(&[0.0; 4], &[2, 2]).with(|s| s.payload_type = Some("bogus")));
    assert!(
        error.starts_with("Unknown payloadType 'bogus'"),
        "{}",
        error
    );
}

#[test]
fn unknown_color_space_is_rejected() {
    let error = run_error(Spec::f32s(&[0.0; 12], &[2, 2, 3]).with(|s| {
        s.payload_type = Some("image");
        s.color_space = Some("cmyk");
    }));
    assert!(error.starts_with("Unknown color space 'cmyk'"), "{}", error);
}

#[test]
fn image_color_space_must_be_inferable() {
    let error = run_error(Spec::f32s(&[0.0; 4], &[2, 2]).with(|s| s.payload_type = Some("image")));
    assert!(error.contains("could not infer a color space"), "{}", error);
}

#[test]
fn layouts_are_validated() {
    let image = run_error(Spec::f32s(&[0.0; 12], &[2, 2, 3]).with(|s| {
        s.payload_type = Some("image");
        s.layout = Some("diagonal");
    }));
    assert!(image.starts_with("Unknown image layout"), "{}", image);
    let audio = run_error(Spec::f32s(&[0.0; 4], &[2, 2]).with(|s| {
        s.payload_type = Some("audio");
        s.layout = Some("diagonal");
    }));
    assert!(audio.starts_with("Unknown audio layout"), "{}", audio);
}

#[test]
fn composite_components_are_validated() {
    let error = pipeline(ECHO_COMPOSITE)
        .run(Some(Spec::composite(vec![Spec::tensor(
            vec![0; 3],
            &[1],
            "f32",
        )])))
        .err()
        .unwrap();
    assert!(error.starts_with("Data buffer size"), "{}", error);
}

#[test]
fn invalid_text_and_unknown_kinds_are_rejected() {
    let p = pipeline(ECHO_BYTES);
    let text = Spec {
        kind: MORFLOW_VALUE_TEXT,
        data: vec![0xff, 0xfe],
        ..Spec::default()
    };
    assert_eq!(
        p.run_args(&[text]).err().unwrap(),
        "String argument is not valid UTF-8"
    );
    let unknown = Spec {
        kind: 99,
        ..Spec::default()
    };
    assert_eq!(
        p.run_args(&[unknown]).err().unwrap(),
        "Unknown value kind 99"
    );
}

// ---------------------------------------------------------------------------
// Asynchronous runs

#[test]
fn async_runs_deliver_through_the_callback() {
    let p = pipeline(ECHO);
    let mut arena = Arena::default();
    let input = arena.value(&Spec::f32s(&[10.0, 20.0, 30.0], &[3]));
    let id = next_request();
    let started = unsafe { morflow_run_async(p.0, &input, 0, deliver, id) };
    assert!(started.is_null());
    drop(arena); // inputs are converted before the call returns
    let out = unsafe { single(wait_for(id)) }.unwrap();
    assert_eq!(out.f32s(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn async_args_and_all_outputs() {
    let p = pipeline("accept Tensor $t\n$t[0:1] >> emit(\"a\")\n$t[1:2] >> emit(\"b\")");
    let mut arena = Arena::default();
    let args = arena.list(&[Spec::f32s(&[1.0, 2.0], &[2])]);
    let id = next_request();
    assert!(
        unsafe { morflow_run_args_async(p.0, args.as_ptr(), args.len(), 1, deliver, id) }.is_null()
    );
    let outs = unsafe { named(wait_for(id)) }.unwrap();
    assert_eq!(outs["a"].f32s(), vec![1.0]);
    assert_eq!(outs["b"].f32s(), vec![2.0]);
}

#[test]
fn async_input_errors_return_at_once_without_a_callback() {
    let p = pipeline(ECHO);
    let mut arena = Arena::default();
    let input = arena.value(&Spec::tensor(vec![0; 8], &[2], "f16"));
    let id = next_request();
    let error = unsafe { single(morflow_run_async(p.0, &input, 0, deliver, id)) }
        .err()
        .unwrap();
    assert!(error.starts_with("Unknown dtype"), "{}", error);
    std::thread::sleep(Duration::from_millis(50));
    assert!(!was_delivered(id));
}

#[test]
fn async_execution_errors_arrive_through_the_callback() {
    let p = pipeline("accept Tensor $t\n$t[0:1] >> emit(\"a\")\n$t[1:2] >> emit(\"b\")");
    let mut arena = Arena::default();
    let input = arena.value(&Spec::f32s(&[1.0, 2.0], &[2]));
    let id = next_request();
    assert!(unsafe { morflow_run_async(p.0, &input, 0, deliver, id) }.is_null());
    let error = unsafe { single(wait_for(id)) }.err().unwrap();
    assert!(
        error.contains("Expected a single emitted output"),
        "{}",
        error
    );
}

#[test]
fn many_concurrent_runs_share_one_pipeline() {
    let p = pipeline(ECHO);
    let mut arena = Arena::default();
    let ids: Vec<(i64, f32)> = (0..32)
        .map(|i| {
            let input = arena.value(&Spec::f32s(&[i as f32], &[1]));
            let id = next_request();
            assert!(unsafe { morflow_run_async(p.0, &input, 0, deliver, id) }.is_null());
            (id, i as f32)
        })
        .collect();
    // Synchronous runs from other threads at the same time.
    std::thread::scope(|scope| {
        for t in 0..4 {
            let p = &p;
            scope.spawn(move || {
                for i in 0..16 {
                    let value = (t * 100 + i) as f32;
                    assert_eq!(
                        p.run(Some(Spec::f32s(&[value], &[1]))).unwrap().f32s(),
                        vec![value]
                    );
                }
            });
        }
    });
    for (id, value) in ids {
        assert_eq!(unsafe { single(wait_for(id)) }.unwrap().f32s(), vec![value]);
    }
}

#[test]
fn freeing_a_pipeline_does_not_cancel_its_runs() {
    let p = pipeline(ECHO);
    let mut arena = Arena::default();
    let input = arena.value(&Spec::f32s(&[5.0], &[1]));
    let id = next_request();
    assert!(unsafe { morflow_run_async(p.0, &input, 0, deliver, id) }.is_null());
    drop(p);
    assert_eq!(unsafe { single(wait_for(id)) }.unwrap().f32s(), vec![5.0]);
}

// ---------------------------------------------------------------------------
// With actions (same pipelines as the Node.js tests)

#[test]
fn identity_action_round_trips() {
    if !actions_available() {
        return;
    }
    let p = pipeline("accept Tensor $tensor\n$tensor >> base/latest/identity >> emit");
    let out = p
        .run(Some(Spec::f32s(&[1.0, 2.5, -3.0, 4.25], &[4])))
        .unwrap();
    assert_eq!(out.f32s(), vec![1.0, 2.5, -3.0, 4.25]);
}

#[test]
fn audio_payloads_encode_to_wav() {
    if !actions_available() {
        return;
    }
    let p = pipeline("import audio_basics/latest\naccept Audio $audio\n$audio >> to_wav >> emit");
    let audio = Spec::f32s(&[0.0; 2000], &[2, 1000]).with(|s| {
        s.payload_type = Some("audio");
        s.sample_rate = Some(48000);
    });
    let wav = p.run(Some(audio)).unwrap();
    assert_eq!(&wav.data()[0..4], b"RIFF");
    assert_eq!(&wav.data()[8..12], b"WAVE");
    // A bare [2, N] tensor is not audio.
    let error = p
        .run(Some(Spec::f32s(&[0.0; 2000], &[2, 1000])))
        .err()
        .unwrap();
    assert!(error.contains("expected Audio"), "{}", error);
}

#[test]
fn image_payloads_reach_image_actions() {
    if !actions_available() {
        return;
    }
    let p = pipeline("from base/latest import to_tensor\nfrom image_basics/latest import to_image\naccept Image $image\n$image >> to_tensor >> to_image >> emit");
    let image = Spec::f32s(&[0.0; 64], &[8, 8, 1]).with(|s| {
        s.payload_type = Some("image");
        s.color_space = Some("grayscale");
    });
    assert_eq!(p.run(Some(image)).unwrap().shape(), &[8, 8]);
}

#[test]
fn int_arg_defaults_and_overrides() {
    if !actions_available() {
        return;
    }
    let p = pipeline("import audio_basics/latest\naccept Bytes $data\naccept IntArg $rate = 48000\n$data >> to_audio(channels=2, sample_rate=$rate, dtype=\"i16\") >> to_wav >> emit");
    for args in [
        vec![Spec::bytes(&[0; 8])],
        vec![Spec::bytes(&[0; 8]), Spec::number(22050.0)],
    ] {
        assert_eq!(&p.run_args(&args).unwrap().data()[0..4], b"RIFF");
    }
}

#[test]
fn qr_composite_components_are_selectable() {
    if !actions_available() {
        return;
    }
    let p = pipeline("import linalg_basics/latest\naccept Tensor[3,2] $matrix\n$matrix >> qr >> $parts\n$parts[0] >> emit(\"q\")\n$parts[1] >> emit(\"r\")\n$parts >> matmul >> emit(\"reconstructed\")");
    let data = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let outs = p.run_all(Spec::f32s(&data, &[3, 2])).unwrap();
    assert_eq!(outs["q"].shape(), &[3, 2]);
    assert_eq!(outs["r"].shape(), &[2, 2]);
    for (got, want) in outs["reconstructed"].f32s().iter().zip(data) {
        assert!((got - want).abs() < 1e-5);
    }
}

#[test]
fn integer_sum_returns_i64() {
    if !actions_available() {
        return;
    }
    let p =
        pipeline("import tensor_stats/latest\naccept Tensor $data\n$data >> sum(axis=0) >> emit");
    let data = [1i32, 2, 3, 4]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let out = p.run(Some(Spec::tensor(data, &[2, 2], "i32"))).unwrap();
    assert_eq!(
        out,
        Out::Tensor {
            dtype: "i64".into(),
            shape: vec![2],
            data: [4i64, 6].iter().flat_map(|v| v.to_ne_bytes()).collect()
        }
    );
}
