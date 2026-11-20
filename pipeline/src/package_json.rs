//! Reads and edits a React Native app's `package.json` for the
//! `--prod-target react-native` commands: the packs the app bundles, under
//! `"morflow": { "packs": { "<pack>": "<version>" } }`.
//!
//! The file is parsed keeping its key order and written back with its own
//! indentation, the way npm writes package.json, so only the `morflow` entry
//! changes.
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The package that runs pipelines in a React Native app.
pub const PACKAGE: &str = "react-native-morflow";

/// The packs listed under `morflow.packs`: name -> version.
pub fn packs(text: &str) -> Result<BTreeMap<String, String>, String> {
    let root = parse(text)?;
    let Some(packs) = packs_object(&root)? else {
        return Ok(BTreeMap::new());
    };
    packs
        .iter()
        .map(|(name, version)| match version {
            Value::String(version) => Ok((name.clone(), version.clone())),
            other => Err(format!(
                "morflow.packs.{name} in package.json must be a version string such as \"0.3.4\", not {other}"
            )),
        })
        .collect()
}

/// `text` with `packs` set under `morflow.packs`, keeping other packs.
pub fn update(text: &str, packs: &BTreeMap<String, String>) -> Result<String, String> {
    let mut root = parse(text)?;
    if packs.is_empty() {
        return Ok(text.to_owned());
    }
    let object = root.as_object_mut().expect("parse checks the root");
    let morflow = object
        .entry("morflow")
        .or_insert_with(|| Value::Object(Map::new()));
    let morflow = morflow
        .as_object_mut()
        .ok_or("\"morflow\" in package.json must be an object")?;
    let list = morflow
        .entry("packs")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or("\"morflow.packs\" in package.json must be an object")?;
    let mut changed = false;
    for (name, version) in packs {
        if list.get(name).and_then(Value::as_str) != Some(version) {
            list.insert(name.clone(), Value::String(version.clone()));
            changed = true;
        }
    }
    if !changed {
        return Ok(text.to_owned());
    }
    write(&root, text)
}

/// `text` without the packs `names` (every pack when None). An emptied
/// `packs`, and then an emptied `morflow`, are removed too.
pub fn remove_packs(text: &str, names: Option<&[String]>) -> Result<String, String> {
    let mut root = parse(text)?;
    if packs_object(&root)?.is_none_or(|packs| packs.is_empty() && names.is_some()) {
        return Ok(text.to_owned());
    }
    let object = root.as_object_mut().expect("parse checks the root");
    let morflow = object
        .get_mut("morflow")
        .and_then(Value::as_object_mut)
        .expect("packs_object checks morflow");
    let list = morflow
        .get_mut("packs")
        .and_then(Value::as_object_mut)
        .expect("packs_object checks packs");
    let before = list.len();
    list.retain(|name, _| names.is_some_and(|names| !names.contains(name)));
    if list.len() == before && names.is_some() {
        return Ok(text.to_owned());
    }
    if list.is_empty() {
        morflow.shift_remove("packs");
    }
    if morflow.is_empty() {
        object.shift_remove("morflow");
    }
    write(&root, text)
}

/// Whether `dependencies` names `package`.
pub fn depends_on(text: &str, package: &str) -> bool {
    parse(text).is_ok_and(|root| {
        root.get("dependencies")
            .and_then(Value::as_object)
            .is_some_and(|deps| deps.contains_key(package))
    })
}

fn parse(text: &str) -> Result<Value, String> {
    let root: Value =
        serde_json::from_str(text).map_err(|e| format!("Cannot read package.json: {e}"))?;
    if !root.is_object() {
        return Err("package.json must hold a JSON object".into());
    }
    Ok(root)
}

/// The `morflow.packs` object, if there is one.
fn packs_object(root: &Value) -> Result<Option<&Map<String, Value>>, String> {
    let Some(morflow) = root.get("morflow") else {
        return Ok(None);
    };
    let morflow = morflow
        .as_object()
        .ok_or("\"morflow\" in package.json must be an object")?;
    match morflow.get("packs") {
        None => Ok(None),
        Some(Value::Object(packs)) => Ok(Some(packs)),
        Some(_) => Err("\"morflow.packs\" in package.json must be an object".into()),
    }
}

/// `root` as JSON indented like `original`, ending with a newline if it did.
fn write(root: &Value, original: &str) -> Result<String, String> {
    use serde::Serialize;
    let indent = original
        .lines()
        .skip(1)
        .find(|line| !line.trim().is_empty())
        .map(|line| &line[..line.len() - line.trim_start().len()])
        .filter(|indent| !indent.is_empty())
        .unwrap_or("  ");
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(indent.as_bytes());
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    root.serialize(&mut serializer).map_err(|e| e.to_string())?;
    let mut text = String::from_utf8(out).map_err(|e| e.to_string())?;
    if original.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packs_of(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    const APP: &str = r#"{
  "name": "app",
  "version": "0.0.1",
  "scripts": {
    "android": "react-native run-android"
  },
  "dependencies": {
    "react": "19.2.0",
    "react-native": "0.88.1",
    "react-native-morflow": "^0.3.4"
  }
}
"#;

    #[test]
    fn adds_packs_after_everything_else_and_keeps_the_rest() {
        let after = update(
            APP,
            &packs_of(&[("base", "0.3.4"), ("audio_basics", "0.2.4")]),
        )
        .unwrap();
        assert_eq!(
            after,
            r#"{
  "name": "app",
  "version": "0.0.1",
  "scripts": {
    "android": "react-native run-android"
  },
  "dependencies": {
    "react": "19.2.0",
    "react-native": "0.88.1",
    "react-native-morflow": "^0.3.4"
  },
  "morflow": {
    "packs": {
      "audio_basics": "0.2.4",
      "base": "0.3.4"
    }
  }
}
"#
        );
        assert_eq!(
            packs(&after).unwrap(),
            packs_of(&[("audio_basics", "0.2.4"), ("base", "0.3.4")])
        );
        assert_eq!(
            update(&after, &packs_of(&[("base", "0.3.4")])).unwrap(),
            after
        );
    }

    #[test]
    fn updates_a_version_in_place_and_keeps_the_file_indentation() {
        let before = "{\n    \"name\": \"app\",\n    \"morflow\": {\n        \"packs\": {\n            \"base\": \"0.3.3\",\n            \"zeta\": \"1.0.0\"\n        },\n        \"note\": \"kept\"\n    }\n}";
        let after = update(before, &packs_of(&[("base", "0.3.4")])).unwrap();
        assert_eq!(
            after,
            "{\n    \"name\": \"app\",\n    \"morflow\": {\n        \"packs\": {\n            \"base\": \"0.3.4\",\n            \"zeta\": \"1.0.0\"\n        },\n        \"note\": \"kept\"\n    }\n}"
        );
    }

    #[test]
    fn removing_packs_removes_what_is_left_empty() {
        let full = update(
            APP,
            &packs_of(&[("base", "0.3.4"), ("audio_basics", "0.2.4")]),
        )
        .unwrap();
        let one = remove_packs(&full, Some(&["base".into(), "missing".into()])).unwrap();
        assert_eq!(packs(&one).unwrap(), packs_of(&[("audio_basics", "0.2.4")]));
        assert_eq!(
            remove_packs(&one, Some(&["audio_basics".into()])).unwrap(),
            APP
        );
        assert_eq!(remove_packs(&full, None).unwrap(), APP);
        assert_eq!(remove_packs(APP, None).unwrap(), APP);
        assert_eq!(
            remove_packs(&full, Some(&["missing".into()])).unwrap(),
            full
        );
        // Other settings under morflow stay.
        let other = "{\"morflow\": {\"packs\": {\"base\": \"0.3.4\"}, \"x\": 1}}";
        assert_eq!(
            remove_packs(other, None).unwrap(),
            "{\n  \"morflow\": {\n    \"x\": 1\n  }\n}"
        );
    }

    #[test]
    fn reports_what_it_cannot_read() {
        assert!(packs("{").unwrap_err().contains("Cannot read package.json"));
        assert!(packs("[]").unwrap_err().contains("JSON object"));
        assert!(packs("{\"morflow\": 1}")
            .unwrap_err()
            .contains("must be an object"));
        assert!(packs("{\"morflow\": {\"packs\": {\"base\": 3}}}")
            .unwrap_err()
            .contains("version string"));
        assert!(update(
            "{\"morflow\": {\"packs\": []}}",
            &packs_of(&[("base", "0.3.4")])
        )
        .unwrap_err()
        .contains("must be an object"));
    }

    #[test]
    fn finds_the_package_dependency() {
        assert!(depends_on(APP, PACKAGE));
        assert!(!depends_on(
            "{\"devDependencies\": {\"react-native-morflow\": \"1\"}}",
            PACKAGE
        ));
        assert!(!depends_on("not json", PACKAGE));
    }
}
