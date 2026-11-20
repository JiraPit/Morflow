// The Morflow C interface (bindings/capi/src/lib.rs), loaded at runtime from
// the libmorflow_capi library the app bundles. The structs must match the
// #[repr(C)] structs there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace morflow {

constexpr uint32_t kValueTensor = 0;
constexpr uint32_t kValueBytes = 1;
constexpr uint32_t kValueNumber = 2;
constexpr uint32_t kValueText = 3;
constexpr uint32_t kValueBool = 4;

struct ValueIn {
  uint32_t kind;
  const uint8_t *data;
  size_t data_len;
  const uint64_t *shape;
  size_t rank;
  const char *dtype;
  const char *payload_type;
  const char *color_space;
  const char *layout;
  int64_t sample_rate;
  int64_t channels;
  const ValueIn *items;
  size_t item_count;
  double number;
  uint8_t flag;
};

struct ValueOut {
  const char *dtype;
  const uint64_t *shape;
  size_t rank;
  const uint8_t *data;
  size_t data_len;
  const ValueOut *items;
  size_t item_count;
};

struct Pipeline;
struct Result;
using Callback = void (*)(int64_t request_id, Result *result);

// The C interface's functions.
struct Capi {
  const char *(*version)();
  Pipeline *(*load)(const char *path, char **error);
  Pipeline *(*from_str)(const char *source, char **error);
  void (*pipeline_free)(Pipeline *pipeline);
  size_t (*param_count)(const Pipeline *pipeline);
  char *(*param_name)(const Pipeline *pipeline, size_t index);
  void (*string_free)(char *text);
  Result *(*run)(const Pipeline *pipeline, const ValueIn *input, uint8_t all);
  Result *(*run_args)(const Pipeline *pipeline, const ValueIn *args, size_t count, uint8_t all);
  Result *(*run_async)(const Pipeline *pipeline, const ValueIn *input, uint8_t all, Callback callback,
                       int64_t request_id);
  Result *(*run_args_async)(const Pipeline *pipeline, const ValueIn *args, size_t count, uint8_t all,
                            Callback callback, int64_t request_id);
  const char *(*result_error)(const Result *result);
  size_t (*result_count)(const Result *result);
  const char *(*result_name)(const Result *result, size_t index);
  const ValueOut *(*result_value)(const Result *result, size_t index);
  void (*result_free)(Result *result);
};

// The C interface, loaded on first use. Null with `error` set when the
// library or one of its functions is missing.
const Capi *capi(std::string &error);

} // namespace morflow
