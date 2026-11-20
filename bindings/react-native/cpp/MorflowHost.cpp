#include "MorflowHost.h"

#include "MorflowCapi.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace morflow {

namespace jsi = facebook::jsi;
using facebook::react::CallInvoker;

namespace {

const Capi &api(jsi::Runtime &rt) {
  std::string error;
  const Capi *loaded = capi(error);
  if (!loaded) {
    throw jsi::JSError(rt, "Morflow's native library could not be loaded (" + error +
                               "). Rebuild the app after installing react-native-morflow; in an Expo app, "
                               "use a development build (npx expo run:android or run:ios), not Expo Go.");
  }
  return *loaded;
}

// ---------------------------------------------------------------------------
// Inputs

// Keeps everything a converted input points to alive until the call returns.
// std::deque never moves its elements, so pointers into it stay valid.
struct InputStore {
  std::deque<std::string> strings;
  std::deque<std::vector<uint64_t>> shapes;
  std::deque<std::vector<ValueIn>> lists;
};

ValueIn empty_value() {
  ValueIn value;
  std::memset(&value, 0, sizeof value);
  value.sample_rate = -1;
  value.channels = -1;
  return value;
}

// The bytes of a Uint8Array (or any typed array), without copying.
void bytes_of(jsi::Runtime &rt, const jsi::Value &view, ValueIn &out) {
  if (!view.isObject()) {
    throw jsi::JSError(rt, "Expected a Uint8Array");
  }
  jsi::Object object = view.asObject(rt);
  jsi::Value buffer = object.getProperty(rt, "buffer");
  if (!buffer.isObject() || !buffer.asObject(rt).isArrayBuffer(rt)) {
    throw jsi::JSError(rt, "Expected a Uint8Array");
  }
  jsi::ArrayBuffer array = buffer.asObject(rt).getArrayBuffer(rt);
  auto offset = static_cast<size_t>(object.getProperty(rt, "byteOffset").asNumber());
  auto length = static_cast<size_t>(object.getProperty(rt, "byteLength").asNumber());
  if (offset + length > array.size(rt)) {
    throw jsi::JSError(rt, "Typed array is outside its buffer");
  }
  out.data = array.data(rt) + offset;
  out.data_len = length;
}

const char *optional_string(jsi::Runtime &rt, const jsi::Object &object, const char *name, InputStore &store) {
  jsi::Value value = object.getProperty(rt, name);
  if (value.isUndefined() || value.isNull()) {
    return nullptr;
  }
  if (!value.isString()) {
    throw jsi::JSError(rt, std::string(name) + " must be a string");
  }
  store.strings.push_back(value.asString(rt).utf8(rt));
  return store.strings.back().c_str();
}

int64_t optional_integer(jsi::Runtime &rt, const jsi::Object &object, const char *name) {
  jsi::Value value = object.getProperty(rt, name);
  if (value.isUndefined() || value.isNull()) {
    return -1;
  }
  if (!value.isNumber() || value.asNumber() < 0) {
    throw jsi::JSError(rt, std::string(name) + " must be a non-negative number");
  }
  return static_cast<int64_t>(value.asNumber());
}

ValueIn convert(jsi::Runtime &rt, const jsi::Value &value, InputStore &store) {
  ValueIn out = empty_value();
  if (value.isNumber()) {
    out.kind = kValueNumber;
    out.number = value.asNumber();
    return out;
  }
  if (value.isBool()) {
    out.kind = kValueBool;
    out.flag = value.getBool() ? 1 : 0;
    return out;
  }
  if (value.isString()) {
    store.strings.push_back(value.asString(rt).utf8(rt));
    out.kind = kValueText;
    out.data = reinterpret_cast<const uint8_t *>(store.strings.back().data());
    out.data_len = store.strings.back().size();
    return out;
  }
  if (!value.isObject()) {
    throw jsi::JSError(rt, "Unsupported pipeline input");
  }
  jsi::Object object = value.asObject(rt);
  std::string kind = object.getProperty(rt, "kind").asString(rt).utf8(rt);
  if (kind == "bytes") {
    out.kind = kValueBytes;
    bytes_of(rt, object.getProperty(rt, "data"), out);
    return out;
  }
  out.kind = kValueTensor;
  out.payload_type = optional_string(rt, object, "payloadType", store);
  jsi::Value items = object.getProperty(rt, "items");
  if (items.isObject()) {
    jsi::Array array = items.asObject(rt).asArray(rt);
    std::vector<ValueIn> converted;
    size_t count = array.size(rt);
    converted.reserve(count);
    for (size_t i = 0; i < count; i++) {
      converted.push_back(convert(rt, array.getValueAtIndex(rt, i), store));
    }
    store.lists.push_back(std::move(converted));
    out.items = store.lists.back().data();
    out.item_count = store.lists.back().size();
    return out;
  }
  bytes_of(rt, object.getProperty(rt, "data"), out);
  jsi::Array shape = object.getProperty(rt, "shape").asObject(rt).asArray(rt);
  std::vector<uint64_t> dims;
  size_t rank = shape.size(rt);
  dims.reserve(rank);
  for (size_t i = 0; i < rank; i++) {
    double dim = shape.getValueAtIndex(rt, i).asNumber();
    if (dim < 0) {
      throw jsi::JSError(rt, "shape must hold non-negative numbers");
    }
    dims.push_back(static_cast<uint64_t>(dim));
  }
  store.shapes.push_back(std::move(dims));
  out.shape = store.shapes.back().data();
  out.rank = store.shapes.back().size();
  out.dtype = optional_string(rt, object, "dtype", store);
  out.color_space = optional_string(rt, object, "colorSpace", store);
  out.layout = optional_string(rt, object, "layout", store);
  out.sample_rate = optional_integer(rt, object, "sampleRate");
  out.channels = optional_integer(rt, object, "channels");
  return out;
}

// ---------------------------------------------------------------------------
// Outputs

class OwnedBuffer : public jsi::MutableBuffer {
 public:
  OwnedBuffer(const uint8_t *data, size_t size) : bytes_(data, data + size) {}
  size_t size() const override {
    return bytes_.size();
  }
  uint8_t *data() override {
    return bytes_.data();
  }

 private:
  std::vector<uint8_t> bytes_;
};

jsi::Value output(jsi::Runtime &rt, const ValueOut &value) {
  jsi::Object out(rt);
  out.setProperty(rt, "dtype", jsi::String::createFromUtf8(rt, value.dtype ? value.dtype : ""));
  if (value.dtype && std::strcmp(value.dtype, "composite") == 0) {
    jsi::Array items(rt, value.item_count);
    for (size_t i = 0; i < value.item_count; i++) {
      items.setValueAtIndex(rt, i, output(rt, value.items[i]));
    }
    out.setProperty(rt, "items", items);
    return out;
  }
  jsi::Array shape(rt, value.rank);
  for (size_t i = 0; i < value.rank; i++) {
    shape.setValueAtIndex(rt, i, static_cast<double>(value.shape[i]));
  }
  out.setProperty(rt, "shape", shape);
  out.setProperty(rt, "data", jsi::ArrayBuffer(rt, std::make_shared<OwnedBuffer>(value.data, value.data_len)));
  return out;
}

// The result as JavaScript, or the error message. Frees the result.
jsi::Value take_result(jsi::Runtime &rt, const Capi &c, Result *result, bool all, std::string &error) {
  struct Free {
    const Capi &c;
    Result *r;
    ~Free() {
      c.result_free(r);
    }
  } free_result{c, result};
  if (const char *message = c.result_error(result)) {
    error = message;
    return jsi::Value::undefined();
  }
  size_t count = c.result_count(result);
  if (!all) {
    return count > 0 ? output(rt, *c.result_value(result, 0)) : jsi::Value::undefined();
  }
  jsi::Object outputs(rt);
  for (size_t i = 0; i < count; i++) {
    const char *name = c.result_name(result, i);
    outputs.setProperty(rt, name ? name : "", output(rt, *c.result_value(result, i)));
  }
  return outputs;
}

jsi::Value finish_sync(jsi::Runtime &rt, const Capi &c, Result *result, bool all) {
  std::string error;
  jsi::Value value = take_result(rt, c, result, all, error);
  if (!error.empty()) {
    throw jsi::JSError(rt, error);
  }
  return value;
}

// ---------------------------------------------------------------------------
// Async runs

// The Promise functions of a run in flight. Created and deleted only on the
// JavaScript thread; leaked if the runtime goes away first, since jsi values
// cannot be destroyed without it.
struct Settle {
  jsi::Function resolve;
  jsi::Function reject;
};

struct Pending {
  std::shared_ptr<CallInvoker> invoker;
  std::shared_ptr<std::atomic<bool>> alive;
  Settle *settle;
  bool all;
};

std::mutex pending_mutex;
std::unordered_map<int64_t, Pending> pending;
std::atomic<int64_t> next_request{1};

void reject(jsi::Runtime &rt, Settle &settle, const std::string &message) {
  jsi::Object error = rt.global().getPropertyAsFunction(rt, "Error").callAsConstructor(rt, message).asObject(rt);
  settle.reject.call(rt, error);
}

// Called by the engine on a worker thread when a run finishes.
void on_done(int64_t request_id, Result *result) {
  Pending run;
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    auto found = pending.find(request_id);
    if (found == pending.end()) {
      return;
    }
    run = found->second;
    pending.erase(found);
  }
  run.invoker->invokeAsync([run, result](jsi::Runtime &rt) {
    std::string error;
    const Capi *c = capi(error);
    if (!run.alive->load()) {
      c->result_free(result);
      return;
    }
    Settle *settle = run.settle;
    try {
      jsi::Value value = take_result(rt, *c, result, run.all, error);
      if (error.empty()) {
        settle->resolve.call(rt, value);
      } else {
        reject(rt, *settle, error);
      }
    } catch (const jsi::JSError &e) {
      reject(rt, *settle, e.getMessage());
    }
    delete settle;
  });
}

// A Promise of running `pipeline` with `input` (or `args` when `with_args`).
jsi::Value run_async(jsi::Runtime &rt, Pipeline *pipeline, bool all, const jsi::Value &input, bool with_args,
                     const std::shared_ptr<CallInvoker> &invoker, const std::shared_ptr<std::atomic<bool>> &alive) {
  const Capi &c = api(rt);
  // The input is converted, and the run started, inside the Promise
  // executor, which runs before the Promise constructor returns.
  auto executor = jsi::Function::createFromHostFunction(
      rt, jsi::PropNameID::forAscii(rt, "executor"), 2,
      [&c, pipeline, all, &input, with_args, invoker, alive](jsi::Runtime &rt, const jsi::Value &,
                                                              const jsi::Value *args, size_t) -> jsi::Value {
        auto *settle = new Settle{args[0].asObject(rt).asFunction(rt), args[1].asObject(rt).asFunction(rt)};
        int64_t id = next_request++;
        {
          std::lock_guard<std::mutex> lock(pending_mutex);
          pending[id] = Pending{invoker, alive, settle, all};
        }
        Result *failed = nullptr;
        try {
          InputStore store;
          if (with_args) {
            jsi::Array list = input.asObject(rt).asArray(rt);
            std::vector<ValueIn> values;
            for (size_t i = 0; i < list.size(rt); i++) {
              values.push_back(convert(rt, list.getValueAtIndex(rt, i), store));
            }
            failed = c.run_args_async(pipeline, values.data(), values.size(), all ? 1 : 0, on_done, id);
          } else if (input.isUndefined() || input.isNull()) {
            failed = c.run_async(pipeline, nullptr, all ? 1 : 0, on_done, id);
          } else {
            ValueIn value = convert(rt, input, store);
            failed = c.run_async(pipeline, &value, all ? 1 : 0, on_done, id);
          }
        } catch (const jsi::JSError &e) {
          std::lock_guard<std::mutex> lock(pending_mutex);
          pending.erase(id);
          reject(rt, *settle, e.getMessage());
          delete settle;
          return jsi::Value::undefined();
        }
        if (failed) {
          // The run did not start, so the callback never fires.
          {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending.erase(id);
          }
          std::string error;
          take_result(rt, c, failed, all, error);
          reject(rt, *settle, error);
          delete settle;
        }
        return jsi::Value::undefined();
      });
  return rt.global().getPropertyAsFunction(rt, "Promise").callAsConstructor(rt, executor);
}

// ---------------------------------------------------------------------------
// Pipelines

class PipelineHost : public jsi::HostObject {
 public:
  PipelineHost(Pipeline *pipeline, std::shared_ptr<CallInvoker> invoker, std::shared_ptr<std::atomic<bool>> alive)
      : pipeline_(pipeline), invoker_(std::move(invoker)), alive_(std::move(alive)) {}

  ~PipelineHost() override {
    close();
  }

  jsi::Value get(jsi::Runtime &rt, const jsi::PropNameID &prop) override {
    std::string name = prop.utf8(rt);
    if (name == "params") {
      const Capi &c = api(rt);
      Pipeline *pipeline = open(rt);
      size_t count = c.param_count(pipeline);
      jsi::Array params(rt, count);
      for (size_t i = 0; i < count; i++) {
        char *param = c.param_name(pipeline, i);
        params.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, param ? param : ""));
        c.string_free(param);
      }
      return params;
    }
    if (name == "run" || name == "runArgs" || name == "runAsync" || name == "runArgsAsync") {
      bool with_args = name == "runArgs" || name == "runArgsAsync";
      bool async = name == "runAsync" || name == "runArgsAsync";
      return jsi::Function::createFromHostFunction(
          rt, prop, 2,
          [this, with_args, async](jsi::Runtime &rt, const jsi::Value &, const jsi::Value *args,
                                   size_t count) -> jsi::Value {
            bool all = count > 0 && args[0].isBool() && args[0].getBool();
            jsi::Value none;
            const jsi::Value &input = count > 1 ? args[1] : none;
            Pipeline *pipeline = open(rt);
            if (async) {
              return run_async(rt, pipeline, all, input, with_args, invoker_, alive_);
            }
            const Capi &c = api(rt);
            InputStore store;
            if (with_args) {
              jsi::Array list = input.asObject(rt).asArray(rt);
              std::vector<ValueIn> values;
              for (size_t i = 0; i < list.size(rt); i++) {
                values.push_back(convert(rt, list.getValueAtIndex(rt, i), store));
              }
              return finish_sync(rt, c, c.run_args(pipeline, values.data(), values.size(), all ? 1 : 0), all);
            }
            if (input.isUndefined() || input.isNull()) {
              return finish_sync(rt, c, c.run(pipeline, nullptr, all ? 1 : 0), all);
            }
            ValueIn value = convert(rt, input, store);
            return finish_sync(rt, c, c.run(pipeline, &value, all ? 1 : 0), all);
          });
    }
    if (name == "close") {
      return jsi::Function::createFromHostFunction(
          rt, prop, 0, [this](jsi::Runtime &, const jsi::Value &, const jsi::Value *, size_t) -> jsi::Value {
            close();
            return jsi::Value::undefined();
          });
    }
    return jsi::Value::undefined();
  }

 private:
  Pipeline *open(jsi::Runtime &rt) {
    if (!pipeline_) {
      throw jsi::JSError(rt, "This pipeline was closed");
    }
    return pipeline_;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pipeline_) {
      std::string error;
      if (const Capi *c = capi(error)) {
        // Runs in flight hold their own copy of the pipeline.
        c->pipeline_free(pipeline_);
      }
      pipeline_ = nullptr;
    }
  }

  Pipeline *pipeline_;
  std::shared_ptr<CallInvoker> invoker_;
  std::shared_ptr<std::atomic<bool>> alive_;
  std::mutex mutex_;
};

jsi::Value compile(jsi::Runtime &rt, bool from_path, const jsi::Value *args, size_t count,
                   const std::shared_ptr<CallInvoker> &invoker, const std::shared_ptr<std::atomic<bool>> &alive) {
  const Capi &c = api(rt);
  if (count < 1 || !args[0].isString()) {
    throw jsi::JSError(rt, from_path ? "load() takes a file path" : "fromStr() takes the pipeline's source text");
  }
  std::string text = args[0].asString(rt).utf8(rt);
  char *error = nullptr;
  Pipeline *pipeline = from_path ? c.load(text.c_str(), &error) : c.from_str(text.c_str(), &error);
  if (!pipeline) {
    std::string message = error ? error : "The pipeline could not be compiled";
    c.string_free(error);
    throw jsi::JSError(rt, message);
  }
  return jsi::Object::createFromHostObject(rt, std::make_shared<PipelineHost>(pipeline, invoker, alive));
}

} // namespace

void install(jsi::Runtime &rt, std::shared_ptr<CallInvoker> invoker, std::shared_ptr<std::atomic<bool>> alive) {
  jsi::Object morflow(rt);
  morflow.setProperty(rt, "version",
                      jsi::Function::createFromHostFunction(
                          rt, jsi::PropNameID::forAscii(rt, "version"), 0,
                          [](jsi::Runtime &rt, const jsi::Value &, const jsi::Value *, size_t) -> jsi::Value {
                            return jsi::String::createFromUtf8(rt, api(rt).version());
                          }));
  morflow.setProperty(rt, "fromStr",
                      jsi::Function::createFromHostFunction(
                          rt, jsi::PropNameID::forAscii(rt, "fromStr"), 1,
                          [invoker, alive](jsi::Runtime &rt, const jsi::Value &, const jsi::Value *args,
                                           size_t count) -> jsi::Value {
                            return compile(rt, false, args, count, invoker, alive);
                          }));
  morflow.setProperty(rt, "load",
                      jsi::Function::createFromHostFunction(
                          rt, jsi::PropNameID::forAscii(rt, "load"), 1,
                          [invoker, alive](jsi::Runtime &rt, const jsi::Value &, const jsi::Value *args,
                                           size_t count) -> jsi::Value {
                            return compile(rt, true, args, count, invoker, alive);
                          }));
  rt.global().setProperty(rt, "__morflow", morflow);
}

} // namespace morflow
