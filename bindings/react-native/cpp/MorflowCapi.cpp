#include "MorflowCapi.h"

#include <dlfcn.h>
#include <mutex>

namespace morflow {

namespace {

// Where the app bundles the C interface: among the app's native libraries on
// Android, and as a framework in the app's Frameworks directory on iOS. The
// engine finds the pack libraries in the same places (pipeline/src/bundled.rs).
void *open_library(std::string &error) {
#if defined(__ANDROID__)
  const char *candidates[] = {"libmorflow_capi.so"};
#elif defined(__APPLE__)
  const char *candidates[] = {"@rpath/morflow_capi.framework/morflow_capi",
                              "@executable_path/Frameworks/morflow_capi.framework/morflow_capi"};
#else
  const char *candidates[] = {"libmorflow_capi.so"};
#endif
  for (const char *path : candidates) {
    if (void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
      return library;
    }
    const char *reason = dlerror();
    error = reason ? reason : path;
  }
  return nullptr;
}

template <typename T>
bool bind(void *library, const char *name, T &slot, std::string &error) {
  slot = reinterpret_cast<T>(dlsym(library, name));
  if (!slot) {
    error = std::string("libmorflow_capi has no ") + name + "; it is not the version this package needs";
  }
  return slot != nullptr;
}

} // namespace

const Capi *capi(std::string &error) {
  static std::once_flag once;
  static Capi api{};
  static const Capi *loaded = nullptr;
  static std::string load_error;
  std::call_once(once, [] {
    void *library = open_library(load_error);
    if (!library) {
      return;
    }
    std::string &e = load_error;
    bool ok = bind(library, "morflow_version", api.version, e) &&
              bind(library, "morflow_load", api.load, e) &&
              bind(library, "morflow_from_str", api.from_str, e) &&
              bind(library, "morflow_pipeline_free", api.pipeline_free, e) &&
              bind(library, "morflow_param_count", api.param_count, e) &&
              bind(library, "morflow_param_name", api.param_name, e) &&
              bind(library, "morflow_string_free", api.string_free, e) &&
              bind(library, "morflow_run", api.run, e) &&
              bind(library, "morflow_run_args", api.run_args, e) &&
              bind(library, "morflow_run_async", api.run_async, e) &&
              bind(library, "morflow_run_args_async", api.run_args_async, e) &&
              bind(library, "morflow_result_error", api.result_error, e) &&
              bind(library, "morflow_result_count", api.result_count, e) &&
              bind(library, "morflow_result_name", api.result_name, e) &&
              bind(library, "morflow_result_value", api.result_value, e) &&
              bind(library, "morflow_result_free", api.result_free, e);
    if (ok) {
      loaded = &api;
    }
  });
  if (!loaded) {
    error = load_error;
  }
  return loaded;
}

} // namespace morflow
