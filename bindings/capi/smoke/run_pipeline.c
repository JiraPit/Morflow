// Loads a built C interface library and runs a pipeline through it:
//
//   cc -o run_pipeline bindings/capi/smoke/run_pipeline.c
//   ./run_pipeline path/to/libmorflow_capi.dylib
//
// CI runs it on the Intel slice of a universal macOS app (under Rosetta), the
// one build of the library no test runs otherwise. The pipeline uses
// math_basics, so the engine must find that pack library bundled in the app.
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Must match the #[repr(C)] structs in bindings/capi/src/lib.rs.
typedef struct {
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
  const void *items;
  size_t item_count;
  double number;
  uint8_t flag;
} ValueIn;

typedef struct {
  const char *dtype;
  const uint64_t *shape;
  size_t rank;
  const uint8_t *data;
  size_t data_len;
  const void *items;
  size_t item_count;
} ValueOut;

static void *symbol(void *library, const char *name) {
  void *found = dlsym(library, name);
  if (!found) {
    fprintf(stderr, "missing symbol %s\n", name);
  }
  return found;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <libmorflow_capi>\n", argv[0]);
    return 2;
  }
  void *library = dlopen(argv[1], RTLD_NOW);
  if (!library) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  const char *(*version)(void) = symbol(library, "morflow_version");
  void *(*from_str)(const char *, char **) = symbol(library, "morflow_from_str");
  void *(*run)(void *, const ValueIn *, uint8_t) = symbol(library, "morflow_run");
  const char *(*result_error)(void *) = symbol(library, "morflow_result_error");
  const ValueOut *(*result_value)(void *, size_t) = symbol(library, "morflow_result_value");
  void (*result_free)(void *) = symbol(library, "morflow_result_free");
  void (*pipeline_free)(void *) = symbol(library, "morflow_pipeline_free");
  if (!version || !from_str || !run || !result_error || !result_value || !result_free || !pipeline_free) {
    return 1;
  }
  printf("engine %s\n", version());

  char *error = NULL;
  void *pipeline = from_str("import math_basics/latest\naccept Tensor $t\n$t >> abs >> emit", &error);
  if (!pipeline) {
    fprintf(stderr, "compile failed: %s\n", error ? error : "(no message)");
    return 1;
  }

  const float input[3] = {-1.5f, 2.0f, -3.0f};
  const uint64_t shape[1] = {3};
  ValueIn value;
  memset(&value, 0, sizeof value);
  value.kind = 0;
  value.data = (const uint8_t *)input;
  value.data_len = sizeof input;
  value.shape = shape;
  value.rank = 1;
  value.dtype = "f32";
  value.sample_rate = -1;
  value.channels = -1;

  void *result = run(pipeline, &value, 0);
  if (result_error(result)) {
    fprintf(stderr, "run failed: %s\n", result_error(result));
    return 1;
  }
  const ValueOut *output = result_value(result, 0);
  if (strcmp(output->dtype, "f32") != 0 || output->data_len != sizeof input) {
    fprintf(stderr, "unexpected output: dtype %s, %zu bytes\n", output->dtype, output->data_len);
    return 1;
  }
  float values[3];
  memcpy(values, output->data, sizeof values);
  printf("abs: %g %g %g\n", values[0], values[1], values[2]);
  int ok = fabsf(values[0] - 1.5f) < 1e-6f && fabsf(values[1] - 2.0f) < 1e-6f && fabsf(values[2] - 3.0f) < 1e-6f;
  result_free(result);
  pipeline_free(pipeline);
  return ok ? 0 : 1;
}
