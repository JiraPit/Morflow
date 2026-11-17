//! Pipelines running actions from bundled pack libraries instead of the
//! action cache. Needs MORFLOW_PACKS_PATH pointing at pack libraries for every
//! plugin-free pack, built by scripts/build_pack.py; skips otherwise.

use std::sync::Once;

use morflow::{ActionRegistry, Morflow, Payload, Tensor};

/// Points HOME at an empty directory so the user's action cache cannot
/// serve these actions; every action must come from a pack library.
fn bundled_only() -> bool {
    static ISOLATE: Once = Once::new();
    if std::env::var_os("MORFLOW_PACKS_PATH").is_none() {
        eprintln!("skipped: set MORFLOW_PACKS_PATH to pack libraries for base and audio_basics");
        return false;
    }
    ISOLATE.call_once(|| {
        let home = std::env::temp_dir().join(format!("morflow-bundled-home-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        std::env::set_var("HOME", &home);
        std::env::remove_var("MORFLOW_ACTIONS_PATH");
    });
    true
}

fn f32s(values: &[f32]) -> Payload {
    Payload::Tensor(Tensor::from_f32_slice(values))
}

fn single(source: &str, input: Payload) -> Result<Payload, String> {
    let mut pipeline = Morflow::from_str(source).map_err(|e| e.to_string())?;
    pipeline
        .run(input)
        .map_err(|e| e.to_string())?
        .into_single()
        .map_err(|e| e.to_string())
}

#[test]
fn latest_imports_run_from_the_pack_library() {
    if !bundled_only() {
        return;
    }
    let out = single("accept Tensor $t\n$t >> base/latest/identity >> emit", f32s(&[1.0, 2.5])).unwrap();
    assert_eq!(out.as_tensor().unwrap().to_vec_f32(), vec![1.0, 2.5]);
}

#[test]
fn exact_versions_must_match_the_bundled_pack() {
    if !bundled_only() {
        return;
    }
    let bundled = single("from base/0.3.4 import identity\naccept Tensor $t\n$t >> identity >> emit", f32s(&[4.0]));
    assert_eq!(bundled.unwrap().as_tensor().unwrap().to_vec_f32(), vec![4.0]);
    let error = single("from base/0.0.1 import identity\naccept Tensor $t\n$t >> identity >> emit", f32s(&[4.0]))
        .err()
        .unwrap();
    assert!(error.contains("base/0.0.1/identity"), "{error}");
    assert!(error.contains("is not bundled; this app bundles base 0.3.4"), "{error}");
}

#[test]
fn whole_pack_imports_list_the_bundled_actions() {
    if !bundled_only() {
        return;
    }
    let source = "import audio_basics/latest\naccept Audio $audio\n$audio >> gain(linear=2.0) >> to_wav >> emit";
    let audio = core_types::Audio::new(
        Tensor::from_f32_shape(&[0.0; 200], vec![2, 100]).unwrap(),
        44100,
        core_types::AudioChannelLayout::from_channel_count(2),
        core_types::AudioLayout::Planar,
    )
    .unwrap();
    let wav = single(source, Payload::Audio(audio)).unwrap();
    let bytes = match wav {
        Payload::Data { buffer } => buffer.to_vec(),
        other => panic!("expected WAV bytes, got {other:?}"),
    };
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
}

#[test]
fn missing_actions_and_packs_are_reported() {
    if !bundled_only() {
        return;
    }
    let error = single("from base/latest import nope\naccept Tensor $t\n$t >> nope >> emit", f32s(&[1.0]))
        .err()
        .unwrap();
    assert!(error.contains("nope"), "{error}");
    // Packs that need a plugin are never bundled.
    let error = single("import linalg_blas/latest\naccept Tensor $t\n$t >> matmul >> emit", f32s(&[1.0]))
        .err()
        .unwrap();
    assert!(error.contains("No prepared catalog for linalg_blas/latest"), "{error}");
}

/// Plugin-free packs, all of which must be bundleable.
const PACKS: [&str; 8] = [
    "base",
    "audio_basics",
    "image_basics",
    "linalg_basics",
    "math_basics",
    "nn_basics",
    "tensor_basics",
    "tensor_stats",
];

/// The action names and version of `pack` in the source tree.
fn source_pack(pack: &str) -> (Vec<String>, String) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../actions").join(pack);
    let mut names = Vec::new();
    let mut versions = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let manifest = entry.unwrap().path().join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else { continue };
        // The first match is in [package], which comes first.
        let field = |key: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(&format!("{key} = \"")))
                .and_then(|rest| rest.strip_suffix('"'))
                .map(str::to_string)
        };
        // Like scripts/package_action.py: metadata.morflow.action, else the package name.
        names.push(field("action").or_else(|| field("name")).unwrap());
        versions.insert(field("version").unwrap());
    }
    names.sort();
    assert_eq!(versions.len(), 1, "{pack} has several versions: {versions:?}");
    (names, versions.into_iter().next().unwrap())
}

#[test]
fn every_pack_library_matches_its_source() {
    if !bundled_only() {
        return;
    }
    let registry = ActionRegistry::new(Vec::new());
    for pack in PACKS {
        let (actions, version) = source_pack(pack);
        let bundled = registry
            .catalog(pack, "latest")
            .unwrap_or_else(|e| panic!("{pack} is not bundled: {e}"));
        assert_eq!(bundled, actions, "{pack} actions");
        assert_eq!(registry.catalog(pack, &version).unwrap(), actions, "{pack} {version}");
        let error = registry.catalog(pack, "0.0.1").unwrap_err();
        assert!(error.contains(&format!("this app bundles {pack} {version}")), "{error}");
    }
}
