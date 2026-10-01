use crate::artifact::{component, normalize_version, ActionIdentity};
use parser::ast::ImportStmt;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackImport {
    pub pack: String,
    pub version: String,
    pub alias: Option<String>,
}

/// Import-scoped action resolution. Catalog lookup is supplied by the caller:
/// preparation uses release manifests; execution uses their offline cache.
#[derive(Debug, Clone, Default)]
pub struct ActionResolver {
    pub imported_symbols: HashMap<String, ActionIdentity>,
    pub imported_packages: Vec<PackImport>,
}
impl ActionResolver {
    pub fn from_imports(imports: &[ImportStmt]) -> Result<Self, String> {
        let mut resolver = Self::default();
        let mut aliases = HashSet::new();
        for import in imports {
            match import {
                ImportStmt::Package(p) => {
                    component(&p.package)?;
                    let version = normalize_version(&p.version)?;
                    let namespace = p.alias.as_ref().unwrap_or(&p.package);
                    if namespace == "emit" || !aliases.insert(namespace.clone()) {
                        return Err(format!("Conflicting import namespace '{namespace}'; use distinct aliases for multiple versions"));
                    }
                    resolver.imported_packages.push(PackImport {
                        pack: p.package.clone(),
                        version,
                        alias: p.alias.clone(),
                    });
                }
                ImportStmt::Items(p) => {
                    for item in &p.items {
                        let name = item.alias.as_ref().unwrap_or(&item.name);
                        if name == "emit" || !aliases.insert(name.clone()) {
                            return Err(format!("Conflicting import alias '{name}'"));
                        }
                        resolver.imported_symbols.insert(
                            name.clone(),
                            ActionIdentity::new(&p.package, &p.version, &item.name)?,
                        );
                    }
                }
            }
        }
        Ok(resolver)
    }
    pub fn resolve<F>(&self, call: &str, mut catalog: F) -> Result<ActionIdentity, String>
    where
        F: FnMut(&str, &str) -> Result<Vec<String>, String>,
    {
        if let Some(id) = self.imported_symbols.get(call) {
            return Ok(id.clone());
        }
        // The parser normalizes qualified paths to '/'; version dots stay in
        // a single slash-delimited segment. Accept dotted namespace.action too.
        let normalized = call.replace("::", "/");
        let parts: Vec<_> = if normalized.contains('/') {
            normalized.split('/').collect()
        } else {
            normalized.split('.').collect()
        };
        if parts.len() == 3 {
            return ActionIdentity::new(parts[0], parts[1], parts[2]);
        }
        if parts.len() == 2 {
            let candidates: Vec<_> = self
                .imported_packages
                .iter()
                .filter(|p| p.alias.as_deref().unwrap_or(&p.pack) == parts[0])
                .collect();
            if candidates.len() == 1 {
                let p = candidates[0];
                return ActionIdentity::new(&p.pack, &p.version, parts[1]);
            }
            return Err(format!(
                "'{call}' requires an imported namespace or explicit pack/version/action path"
            ));
        }
        if parts.len() != 1 {
            return Err(format!(
                "Invalid action path '{call}'; use pack/version/action"
            ));
        }
        let mut matches = Vec::new();
        for p in &self.imported_packages {
            if catalog(&p.pack, &p.version)?
                .iter()
                .any(|name| name == call)
            {
                let id = ActionIdentity::new(&p.pack, &p.version, call)?;
                if !matches.contains(&id) {
                    matches.push(id);
                }
            }
        }
        match matches.len() {
            1 => Ok(matches.remove(0)),
            0 => Err(format!("Action '{call}' is not declared by the imports; import it or use pack/version/action")),
            _ => Err(format!("Ambiguous action '{call}'; qualify it with an import alias")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn resolver(src: &str) -> ActionResolver {
        ActionResolver::from_imports(&parser::parse(src).unwrap().imports).unwrap()
    }
    #[test]
    fn versions_aliases_and_explicit_paths_survive_resolution() {
        let r = resolver("import base/0.1.0 as old\nimport base/0.2.0 as new\nfrom image_basics/latest import resize as scale\n");
        assert_eq!(
            r.resolve("old/identity", |_, _| unreachable!())
                .unwrap()
                .version,
            "0.1.0"
        );
        assert_eq!(
            r.resolve("new.identity", |_, _| unreachable!())
                .unwrap()
                .version,
            "0.2.0"
        );
        assert_eq!(
            r.resolve("scale", |_, _| unreachable!()).unwrap().version,
            "latest"
        );
        assert_eq!(
            r.resolve("base/0.3.0/identity", |_, _| unreachable!())
                .unwrap()
                .version,
            "0.3.0"
        );
        assert!(r
            .resolve("identity", |_, _| Ok(vec!["identity".into()]))
            .unwrap_err()
            .contains("Ambiguous"));
    }
    #[test]
    fn rejects_missing_imports_and_conflicting_aliases() {
        assert!(ActionResolver::default()
            .resolve("identity", |_, _| unreachable!())
            .is_err());
        assert!(ActionResolver::default()
            .resolve("base/identity", |_, _| unreachable!())
            .is_err());
        let ast = parser::parse("import base/0.1.0 as x\nfrom base/0.2.0 import identity as x\n")
            .unwrap();
        assert!(ActionResolver::from_imports(&ast.imports).is_err());
        let ast = parser::parse("import base/latest as emit\n").unwrap();
        assert!(ActionResolver::from_imports(&ast.imports).is_err());
        assert!(ActionIdentity::new("../base", "latest", "identity").is_err());
    }
    #[test]
    fn whole_imports_use_version_specific_catalogs() {
        let r = resolver("import base/0.1.0\nimport custom/0.2.0\n");
        let id = r
            .resolve("special", |pack, version| {
                assert_eq!(version, if pack == "base" { "0.1.0" } else { "0.2.0" });
                Ok(if pack == "custom" {
                    vec!["special".into()]
                } else {
                    vec!["identity".into()]
                })
            })
            .unwrap();
        assert_eq!(id.pack, "custom");
    }
}
