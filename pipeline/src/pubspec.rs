//! Edits a Dart or Flutter app's `pubspec.yaml` for `morflow prep --target
//! flutter`: the packs the app bundles (`hooks.user_defines.morflow.packs`)
//! and its pipelines as Flutter assets.
//!
//! The file is edited as text, line by line, so comments, ordering and
//! formatting elsewhere stay exactly as the developer wrote them. Only block
//! style YAML is understood where it is edited; anything else is reported
//! instead of being rewritten.
use std::collections::BTreeMap;

/// The edited file, or the reason it cannot be edited.
pub fn update(
    text: &str,
    packs: &BTreeMap<String, String>,
    assets: &[String],
) -> Result<String, String> {
    if text.contains('\t') {
        return Err("pubspec.yaml is indented with tabs; YAML needs spaces".into());
    }
    let mut doc = Doc::new(text);
    if !packs.is_empty() {
        let hooks = doc.ensure(None, "hooks")?;
        let defines = doc.ensure(Some(hooks), "user_defines")?;
        let morflow = doc.ensure(Some(defines), "morflow")?;
        let list = doc.ensure(Some(morflow), "packs")?;
        doc.set_entries(list, packs)?;
    }
    if !assets.is_empty() {
        let flutter = doc.find(None, "flutter")?.ok_or(
            "pubspec.yaml has no flutter: section; run this in a Flutter app's root directory",
        )?;
        let list = doc.ensure(Some(flutter), "assets")?;
        doc.add_list_items(list, assets)?;
    }
    Ok(doc.text())
}

/// Whether `dependencies:` names `package`.
pub fn depends_on(text: &str, package: &str) -> bool {
    let doc = Doc::new(text);
    doc.locate(None, "dependencies")
        .is_some_and(|deps| doc.locate(Some(deps), package).is_some())
}

/// A unified diff of `before` -> `after` with `context` lines around changes.
pub fn diff(before: &str, after: &str, context: usize) -> String {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    // Longest common subsequence; pubspecs are small.
    let mut lcs = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    // (kind, line in a, line in b, text)
    let mut ops: Vec<(char, usize, usize, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            ops.push((' ', i, j, a[i]));
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || lcs[i + 1][j] >= lcs[i][j + 1]) {
            ops.push(('-', i, j, a[i]));
            i += 1;
        } else {
            ops.push(('+', i, j, b[j]));
            j += 1;
        }
    }
    let changed: Vec<usize> = (0..ops.len()).filter(|&k| ops[k].0 != ' ').collect();
    let mut out = String::new();
    let mut k = 0;
    while k < changed.len() {
        let start = changed[k].saturating_sub(context);
        let mut end = changed[k];
        while k < changed.len() && changed[k] <= end + 2 * context + 1 {
            end = changed[k];
            k += 1;
        }
        let end = (end + context + 1).min(ops.len());
        let hunk = &ops[start..end];
        let old_len = hunk.iter().filter(|o| o.0 != '+').count();
        let new_len = hunk.iter().filter(|o| o.0 != '-').count();
        out += &format!(
            "@@ -{},{old_len} +{},{new_len} @@\n",
            hunk[0].1 + 1,
            hunk[0].2 + 1
        );
        for (kind, _, _, line) in hunk {
            out += &format!("{kind}{line}\n");
        }
    }
    out
}

struct Doc {
    lines: Vec<String>,
    trailing_newline: bool,
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// A line with YAML content, not blank and not only a comment.
fn is_content(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with('#')
}

/// `value` without a trailing ` # comment`, and the comment.
fn split_comment(value: &str) -> (&str, &str) {
    match value.find(" #") {
        Some(at) => (value[..at].trim(), &value[at..]),
        None if value.trim_start().starts_with('#') => ("", value),
        None => (value.trim(), ""),
    }
}

/// The key of a `key: value` line and the rest after the colon.
fn key_of(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('-') || trimmed.starts_with('#') {
        return None;
    }
    let colon = trimmed.find(':')?;
    let rest = &trimmed[colon + 1..];
    if !(rest.is_empty() || rest.starts_with(' ')) {
        return None;
    }
    let key = trimmed[..colon]
        .trim()
        .trim_matches(|c| c == '"' || c == '\'');
    Some((key, rest))
}

impl Doc {
    fn new(text: &str) -> Self {
        Self {
            lines: text.lines().map(str::to_owned).collect(),
            trailing_newline: text.is_empty() || text.ends_with('\n'),
        }
    }

    fn text(&self) -> String {
        let mut text = self.lines.join("\n");
        if self.trailing_newline {
            text.push('\n');
        }
        text
    }

    /// The lines inside `parent`'s block (the whole file for None), and the
    /// line after its last content line, where new children go.
    fn block(&self, parent: Option<usize>) -> (usize, usize, usize) {
        let Some(parent) = parent else {
            // New top-level sections go at the end, after any trailing
            // comments, so they don't split commented-out template sections.
            let last = self
                .lines
                .iter()
                .rposition(|l| !l.trim().is_empty())
                .map_or(0, |i| i + 1);
            return (0, self.lines.len(), last);
        };
        let own = indent(&self.lines[parent]);
        let mut end = self.lines.len();
        let mut insert = parent + 1;
        for (i, line) in self.lines.iter().enumerate().skip(parent + 1) {
            if !is_content(line) {
                continue;
            }
            // A list may sit at its key's own indentation.
            let list_item = indent(line) == own && line.trim_start().starts_with("- ");
            if indent(line) <= own && !list_item {
                end = i;
                break;
            }
            insert = i + 1;
        }
        (parent + 1, end, insert)
    }

    /// The indentation of `parent`'s children.
    fn child_indent(&self, parent: Option<usize>) -> usize {
        let (start, end, _) = self.block(parent);
        match self.lines[start..end].iter().find(|l| is_content(l)) {
            Some(line) => indent(line),
            None => parent.map_or(0, |p| indent(&self.lines[p]) + self.step()),
        }
    }

    /// The file's indentation step: its smallest indentation, or 2.
    fn step(&self) -> usize {
        self.lines
            .iter()
            .filter(|l| is_content(l))
            .map(|l| indent(l))
            .filter(|&i| i > 0)
            .min()
            .unwrap_or(2)
    }

    /// The line of `key` directly under `parent`, whatever its value.
    fn locate(&self, parent: Option<usize>, key: &str) -> Option<usize> {
        let (start, end, _) = self.block(parent);
        let child = self.child_indent(parent);
        (start..end).find(|&i| {
            let line = &self.lines[i];
            is_content(line)
                && indent(line) == child
                && key_of(line).is_some_and(|(found, _)| found == key)
        })
    }

    /// The line of `key` under `parent`, which must hold a block (or nothing).
    fn find(&self, parent: Option<usize>, key: &str) -> Result<Option<usize>, String> {
        let Some(i) = self.locate(parent, key) else {
            return Ok(None);
        };
        let (_, rest) = key_of(&self.lines[i]).expect("located a key");
        let (value, _) = split_comment(rest);
        if !value.is_empty() && value != "null" && value != "~" {
            return Err(format!(
                "{key}: in pubspec.yaml has a value on the same line ({value}); write it as a block, one entry per line"
            ));
        }
        Ok(Some(i))
    }

    /// `key` under `parent`, added at the end of `parent`'s block if missing.
    fn ensure(&mut self, parent: Option<usize>, key: &str) -> Result<usize, String> {
        if let Some(found) = self.find(parent, key)? {
            // `key: null` or `key: ~` becomes an empty block.
            let line = &self.lines[found];
            let (_, rest) = key_of(line).expect("found a key");
            let (_, comment) = split_comment(rest);
            self.lines[found] = format!("{}{key}:{comment}", " ".repeat(indent(line)));
            return Ok(found);
        }
        let (_, _, insert) = self.block(parent);
        let at = if parent.is_none() && insert > 0 {
            // A new top-level section after a blank line.
            self.lines.insert(insert, String::new());
            insert + 1
        } else {
            insert
        };
        let child = self.child_indent(parent);
        self.lines
            .insert(at, format!("{}{key}:", " ".repeat(child)));
        Ok(at)
    }

    /// Sets `name: version` entries under `parent`, keeping other entries.
    fn set_entries(
        &mut self,
        parent: usize,
        entries: &BTreeMap<String, String>,
    ) -> Result<(), String> {
        let child = self.child_indent(Some(parent));
        let (start, end, _) = self.block(Some(parent));
        let mut missing = entries.clone();
        for i in start..end {
            let line = &self.lines[i];
            if !is_content(line) || indent(line) != child {
                continue;
            }
            let Some((name, rest)) = key_of(line) else {
                return Err(format!(
                    "Cannot read line {} of pubspec.yaml: {}",
                    i + 1,
                    line.trim()
                ));
            };
            if let Some(version) = missing.remove(name) {
                let (_, comment) = split_comment(rest);
                self.lines[i] = format!("{}{name}: {version}{comment}", " ".repeat(child));
            }
        }
        let (_, _, insert) = self.block(Some(parent));
        for (offset, (name, version)) in missing.iter().enumerate() {
            self.lines.insert(
                insert + offset,
                format!("{}{name}: {version}", " ".repeat(child)),
            );
        }
        Ok(())
    }

    /// Adds `- item` entries under `parent` unless an entry already covers them.
    fn add_list_items(&mut self, parent: usize, items: &[String]) -> Result<(), String> {
        let (start, end, insert) = self.block(Some(parent));
        let mut existing = Vec::new();
        let mut item_indent = None;
        for line in &self.lines[start..end] {
            let trimmed = line.trim_start();
            if let Some(item) = trimmed.strip_prefix("- ") {
                item_indent.get_or_insert(indent(line));
                let (item, _) = split_comment(item);
                let item = item.strip_prefix("path:").unwrap_or(item).trim();
                existing.push(item.trim_matches(|c| c == '"' || c == '\'').to_owned());
            }
        }
        let item_indent = item_indent.unwrap_or(indent(&self.lines[parent]) + self.step());
        let covered = |item: &str| {
            existing.iter().any(|e| {
                e == item
                    || (e.ends_with('/')
                        && item
                            .strip_prefix(e.as_str())
                            .is_some_and(|rest| !rest.contains('/')))
            })
        };
        let mut added = 0;
        for item in items {
            if covered(item) {
                continue;
            }
            self.lines.insert(
                insert + added,
                format!("{}- {item}", " ".repeat(item_indent)),
            );
            added += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packs(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    const FLUTTER_APP: &str = "name: app
description: An app.

dependencies:
  flutter:
    sdk: flutter
  morflow: ^0.3.3

flutter:
  uses-material-design: true
";

    #[test]
    fn adds_the_whole_hooks_section_and_assets_to_a_new_app() {
        let after = update(
            FLUTTER_APP,
            &packs(&[("base", "0.3.4"), ("audio_basics", "0.2.4")]),
            &["pipelines/denoise.morf".into()],
        )
        .unwrap();
        assert_eq!(
            after,
            "name: app
description: An app.

dependencies:
  flutter:
    sdk: flutter
  morflow: ^0.3.3

flutter:
  uses-material-design: true
  assets:
    - pipelines/denoise.morf

hooks:
  user_defines:
    morflow:
      packs:
        audio_basics: 0.2.4
        base: 0.3.4
"
        );
    }

    #[test]
    fn updates_listed_packs_in_place_and_keeps_comments_and_other_settings() {
        let before = "name: app
hooks:
  user_defines:
    other_package:
      x: 1
    morflow:
      release_url: https://mirror.example.com # our mirror
      packs:
        # Audio first.
        audio_basics: 0.2.3 # pinned
        tensor_stats: '0.4.1'

flutter:
  assets:
  - images/
";
        let after = update(
            before,
            &packs(&[("audio_basics", "0.2.4"), ("base", "0.3.4")]),
            &[],
        )
        .unwrap();
        assert_eq!(
            after,
            "name: app
hooks:
  user_defines:
    other_package:
      x: 1
    morflow:
      release_url: https://mirror.example.com # our mirror
      packs:
        # Audio first.
        audio_basics: 0.2.4 # pinned
        tensor_stats: '0.4.1'
        base: 0.3.4

flutter:
  assets:
  - images/
"
        );
    }

    #[test]
    fn adds_new_sections_after_trailing_comments() {
        let before = "name: app
flutter:
  uses-material-design: true

  # To add assets to your application, add an assets section, like this:
  # assets:
  #   - images/a_dot_burr.jpeg
";
        let after = update(
            before,
            &packs(&[("base", "0.3.4")]),
            &["pipelines/a.morf".into()],
        )
        .unwrap();
        assert_eq!(
            after,
            "name: app
flutter:
  uses-material-design: true
  assets:
    - pipelines/a.morf

  # To add assets to your application, add an assets section, like this:
  # assets:
  #   - images/a_dot_burr.jpeg

hooks:
  user_defines:
    morflow:
      packs:
        base: 0.3.4
"
        );
    }

    #[test]
    fn adds_morflow_beside_other_packages_user_defines() {
        let before = "name: app\nhooks:\n    user_defines:\n        other:\n            x: 1\n";
        let after = update(before, &packs(&[("base", "0.3.4")]), &[]).unwrap();
        assert_eq!(
            after,
            "name: app\nhooks:\n    user_defines:\n        other:\n            x: 1\n        morflow:\n            packs:\n                base: 0.3.4\n"
        );
    }

    #[test]
    fn assets_already_covered_are_not_repeated() {
        let before = "flutter:\n  assets:\n    - pipelines/\n    - path: other/a.morf\n      flavors: [dev]\n";
        let after = update(
            before,
            &BTreeMap::new(),
            &[
                "pipelines/a.morf".into(),
                "other/a.morf".into(),
                "pipelines/deep/b.morf".into(),
            ],
        )
        .unwrap();
        assert_eq!(
            after,
            "flutter:\n  assets:\n    - pipelines/\n    - path: other/a.morf\n      flavors: [dev]\n    - pipelines/deep/b.morf\n"
        );
        assert_eq!(
            update(&after, &BTreeMap::new(), &["pipelines/a.morf".into()]).unwrap(),
            after
        );
    }

    #[test]
    fn running_twice_changes_nothing_the_second_time() {
        let once = update(
            FLUTTER_APP,
            &packs(&[("base", "0.3.4")]),
            &["p.morf".into()],
        )
        .unwrap();
        assert_eq!(
            update(&once, &packs(&[("base", "0.3.4")]), &["p.morf".into()]).unwrap(),
            once
        );
    }

    #[test]
    fn empty_sections_become_blocks() {
        let before = "flutter:\nhooks: ~\n";
        let after = update(before, &packs(&[("base", "0.3.4")]), &["a.morf".into()]).unwrap();
        assert_eq!(after, "flutter:\n  assets:\n    - a.morf\nhooks:\n  user_defines:\n    morflow:\n      packs:\n        base: 0.3.4\n");
    }

    #[test]
    fn refuses_what_it_cannot_edit_safely() {
        let one_line = "hooks:\n  user_defines:\n    morflow:\n      packs: {base: 0.3.3}\n";
        assert!(update(one_line, &packs(&[("base", "0.3.4")]), &[])
            .unwrap_err()
            .contains("packs: in pubspec.yaml has a value on the same line"));
        assert!(update("name: x\n", &BTreeMap::new(), &["a.morf".into()])
            .unwrap_err()
            .contains("no flutter: section"));
        assert!(
            update("hooks:\n\tx: 1\n", &packs(&[("base", "0.3.4")]), &[])
                .unwrap_err()
                .contains("tabs")
        );
    }

    #[test]
    fn finds_the_morflow_dependency() {
        assert!(depends_on(FLUTTER_APP, "morflow"));
        assert!(!depends_on(
            "dependencies:\n  flutter:\n    sdk: flutter\nmorflow: 1\n",
            "morflow"
        ));
    }

    #[test]
    fn diffs_show_changes_with_context() {
        let before = "a\nb\nc\nd\ne\nf\ng\nh\n";
        let after = "a\nb\nc\nd\nE\nf\ng\nh\ni\n";
        assert_eq!(
            diff(before, after, 1),
            "@@ -4,3 +4,3 @@\n d\n-e\n+E\n f\n@@ -8,1 +8,2 @@\n h\n+i\n"
        );
        assert_eq!(diff(before, before, 3), "");
    }
}
