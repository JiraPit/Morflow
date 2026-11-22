//! Reads and edits the JSON files that list an app's packs for the
//! `--prod-target` commands that use JSON:
//! - a React Native app's `package.json`, under
//!   `"morflow": { "packs": { "<pack>": "<version>" } }`;
//! - a Unity project's `ProjectSettings/Morflow.json`, under
//!   `"packs": { "<pack>": "<version>" }`.
//!
//! A file is parsed keeping its key order and written back with its own
//! indentation, the way npm writes package.json, so only the packs change.
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// The package that runs pipelines in a React Native app.
pub const PACKAGE: &str = "react-native-morflow";

/// A JSON file that lists packs, and where in it they are.
pub struct PackList {
    /// The file's name, for messages.
    pub file: &'static str,
    /// The keys leading to the packs object, outermost first.
    pub keys: &'static [&'static str],
}

/// A React Native app's package.json.
pub const PACKAGE_JSON: PackList = PackList {
    file: "package.json",
    keys: &["morflow", "packs"],
};

/// A Unity project's Morflow settings.
pub const UNITY_SETTINGS: PackList = PackList {
    file: "Morflow.json",
    keys: &["packs"],
};

/// The packs in `package.json`: name -> version.
pub fn packs(text: &str) -> Result<BTreeMap<String, String>, String> {
    PACKAGE_JSON.packs(text)
}

/// `package.json` with `packs` added, keeping other packs.
pub fn update(text: &str, packs: &BTreeMap<String, String>) -> Result<String, String> {
    PACKAGE_JSON.update(text, packs)
}

/// `package.json` without the packs `names` (every pack when None).
pub fn remove_packs(text: &str, names: Option<&[String]>) -> Result<String, String> {
    PACKAGE_JSON.remove_packs(text, names)
}

impl PackList {
    /// The packs listed: name -> version.
    pub fn packs(&self, text: &str) -> Result<BTreeMap<String, String>, String> {
        let root = self.parse(text)?;
        let Some(packs) = self.packs_object(&root)? else {
            return Ok(BTreeMap::new());
        };
        packs
            .iter()
            .map(|(name, version)| match version {
                Value::String(version) => Ok((name.clone(), version.clone())),
                other => Err(format!(
                    "{}.{name} in {} must be a version string such as \"0.3.4\", not {other}",
                    self.path(),
                    self.file
                )),
            })
            .collect()
    }

    /// `text` with `packs` set, keeping other packs.
    pub fn update(&self, text: &str, packs: &BTreeMap<String, String>) -> Result<String, String> {
        let mut root = self.parse(text)?;
        if packs.is_empty() {
            return Ok(text.to_owned());
        }
        let mut object = root.as_object_mut().expect("parse checks the root");
        for (depth, key) in self.keys.iter().enumerate() {
            object = object
                .entry(*key)
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .ok_or_else(|| self.not_an_object(depth))?;
        }
        let mut changed = false;
        for (name, version) in packs {
            if object.get(name).and_then(Value::as_str) != Some(version) {
                object.insert(name.clone(), Value::String(version.clone()));
                changed = true;
            }
        }
        if !changed {
            return Ok(text.to_owned());
        }
        write(&root, text)
    }

    /// `text` without the packs `names` (every pack when None). An emptied
    /// packs object, and then each emptied object around it, are removed too.
    pub fn remove_packs(&self, text: &str, names: Option<&[String]>) -> Result<String, String> {
        let mut root = self.parse(text)?;
        let Some(list) = self.packs_object(&root)? else {
            return Ok(text.to_owned());
        };
        let keep = |name: &String| names.is_some_and(|names| !names.contains(name));
        if list.keys().all(keep) && names.is_some() {
            return Ok(text.to_owned());
        }
        remove(
            root.as_object_mut().expect("parse checks the root"),
            self.keys,
            &keep,
        );
        write(&root, text)
    }

    fn parse(&self, text: &str) -> Result<Value, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|e| format!("Cannot read {}: {e}", self.file))?;
        if !root.is_object() {
            return Err(format!("{} must hold a JSON object", self.file));
        }
        Ok(root)
    }

    /// The packs object, if there is one.
    fn packs_object<'a>(&self, root: &'a Value) -> Result<Option<&'a Map<String, Value>>, String> {
        let mut object = root.as_object().expect("parse checks the root");
        for (depth, key) in self.keys.iter().enumerate() {
            match object.get(*key) {
                None => return Ok(None),
                Some(Value::Object(inner)) => object = inner,
                Some(_) => return Err(self.not_an_object(depth)),
            }
        }
        Ok(Some(object))
    }

    /// The dotted path to the packs, such as `morflow.packs`.
    fn path(&self) -> String {
        self.keys.join(".")
    }

    fn not_an_object(&self, depth: usize) -> String {
        format!(
            "\"{}\" in {} must be an object",
            self.keys[..=depth].join("."),
            self.file
        )
    }
}

/// Removes the packs `keep` rejects from the object at `keys` below
/// `object`, then each object on the way that is left empty.
fn remove(object: &mut Map<String, Value>, keys: &[&str], keep: &dyn Fn(&String) -> bool) {
    let Some((key, rest)) = keys.split_first() else {
        object.retain(|name, _| keep(name));
        return;
    };
    if let Some(inner) = object.get_mut(*key).and_then(Value::as_object_mut) {
        remove(inner, rest, keep);
        if inner.is_empty() {
            object.shift_remove(*key);
        }
    }
}

/// Whether `dependencies` names `package`.
pub fn depends_on(text: &str, package: &str) -> bool {
    serde_json::from_str::<Value>(text).is_ok_and(|root| {
        root.get("dependencies")
            .and_then(Value::as_object)
            .is_some_and(|deps| deps.contains_key(package))
    })
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

    #[test]
    fn unity_settings_keep_packs_at_the_top() {
        let empty = "{}\n";
        let after = UNITY_SETTINGS
            .update(empty, &packs_of(&[("audio_basics", "0.2.5")]))
            .unwrap();
        assert_eq!(
            after,
            "{\n  \"packs\": {\n    \"audio_basics\": \"0.2.5\"\n  }\n}\n"
        );
        assert_eq!(
            UNITY_SETTINGS.packs(&after).unwrap(),
            packs_of(&[("audio_basics", "0.2.5")])
        );
        assert_eq!(UNITY_SETTINGS.remove_packs(&after, None).unwrap(), "{}\n");
        assert!(UNITY_SETTINGS
            .packs("{\"packs\": [1]}")
            .unwrap_err()
            .contains("\"packs\" in Morflow.json must be an object"));
        assert!(UNITY_SETTINGS
            .packs("{\"packs\": {\"base\": 1}}")
            .unwrap_err()
            .contains("packs.base in Morflow.json"));
    }
}
