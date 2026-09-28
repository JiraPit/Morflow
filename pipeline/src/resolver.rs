use std::collections::HashMap;
use parser::ast::ImportStmt;

/// Resolves action names in a flow (e.g. `ca`, `resize`, `base.identity`, `audio.gain`)
/// into their canonical ActionPack and action name based on pipeline import statements.
#[derive(Debug, Clone, Default)]
pub struct ActionResolver {
    /// Maps symbol names in flow (e.g. "ca", "resize") -> (package, action_name) (e.g. ("image_essentials", "color_adjust"))
    pub imported_symbols: HashMap<String, (String, String)>,
    /// List of whole-imported packages in order (e.g. [("base", None), ("image_essentials", Some("img"))])
    pub imported_packages: Vec<(String, Option<String>)>,
}

impl ActionResolver {
    /// Constructs an `ActionResolver` from the list of imports declared in a pipeline AST.
    pub fn from_imports(imports: &[ImportStmt]) -> Self {
        let mut resolver = Self::default();
        for imp in imports {
            match imp {
                ImportStmt::Package(pkg) => {
                    resolver
                        .imported_packages
                        .push((pkg.package.clone(), pkg.alias.clone()));
                }
                ImportStmt::Items(items) => {
                    for item in &items.items {
                        let symbol = item.alias.as_ref().unwrap_or(&item.name).clone();
                        resolver
                            .imported_symbols
                            .insert(symbol, (items.package.clone(), item.name.clone()));
                    }
                }
            }
        }
        resolver
    }

    /// Resolves a called action name in a flow (e.g. "ca", "resize", "base.identity", or "audio.gain")
    /// to (Option<pack_name>, target_action_name).
    pub fn resolve(&self, call_name: &str) -> (Option<String>, String) {
        // 1. Check if the call matches an explicitly imported symbol or alias (`from pack import item [as alias]`)
        if let Some((pack, real_action)) = self.imported_symbols.get(call_name) {
            return (Some(pack.clone()), real_action.clone());
        }

        // 2. Check if the call is a qualified name like `image_essentials.resize` or `audio.gain`
        if let Some((prefix, action)) = call_name.split_once('.') {
            // Check if prefix matches a package alias or package name
            for (pkg, alias_opt) in &self.imported_packages {
                if let Some(alias) = alias_opt {
                    if alias == prefix {
                        return (Some(pkg.clone()), action.to_string());
                    }
                }
                if pkg == prefix {
                    return (Some(pkg.clone()), action.to_string());
                }
            }
            // If prefix wasn't in imported packages list, treat prefix directly as pack name
            return (Some(prefix.to_string()), action.to_string());
        }

        // 3. If there are whole-package imports (`import pack.latest`), check if only one pack is imported
        if self.imported_packages.len() == 1 {
            return (Some(self.imported_packages[0].0.clone()), call_name.to_string());
        }

        // 4. Return None for pack, meaning the registry will search imported packages and known packs
        (None, call_name.to_string())
    }
}
