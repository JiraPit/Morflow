//! Pipelines running actions from bundled pack libraries instead of the
//! action cache. Needs MORFLOW_PACKS_PATH pointing at pack libraries for
//! `base` and `audio_basics` built by scripts/build_pack.py; skips otherwise.

use std::sync::Once;

use morflow::{Morflow, Payload, Tensor};

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
    let bundled = single("from base/0.3.3 import identity\naccept Tensor $t\n$t >> identity >> emit", f32s(&[4.0]));
    assert_eq!(bundled.unwrap().as_tensor().unwrap().to_vec_f32(), vec![4.0]);
    let error = single("from base/0.0.1 import identity\naccept Tensor $t\n$t >> identity >> emit", f32s(&[4.0]))
        .err()
        .unwrap();
    assert!(error.contains("base/0.0.1/identity"), "{error}");
    assert!(error.contains("is not bundled; this app bundles base 0.3.3"), "{error}");
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
    let error = single("import tensor_basics/latest\naccept Tensor $t\n$t >> transpose >> emit", f32s(&[1.0]))
        .err()
        .unwrap();
    assert!(error.contains("No prepared catalog for tensor_basics/latest"), "{error}");
}
