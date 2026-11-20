// The engine's functions in the JavaScript runtime, as `global.__morflow`:
//
//   __morflow.version()          -> the engine version, such as "0.3.4"
//   __morflow.fromStr(source)    -> a pipeline
//   __morflow.load(path)         -> a pipeline
//   pipeline.params              -> parameter names, in declaration order
//   pipeline.run(all, input)     -> an output (all: false) or named outputs
//   pipeline.runArgs(all, args)
//   pipeline.runAsync(all, input)      -> a Promise of the same
//   pipeline.runArgsAsync(all, args)
//   pipeline.close()             -> frees the pipeline before garbage collection
//
// Inputs arrive normalized by src/index.ts: a number, string or boolean, or
// { kind: 'bytes', data: Uint8Array } or { kind: 'tensor', data: Uint8Array,
// shape, dtype?, payloadType?, colorSpace?, layout?, sampleRate?, channels?,
// items? }. An output is { dtype, shape, data: ArrayBuffer }, or
// { dtype: 'composite', items }.
#pragma once

#include <ReactCommon/CallInvoker.h>
#include <jsi/jsi.h>

#include <atomic>
#include <memory>

namespace morflow {

// Adds `global.__morflow`. Async results are delivered on the JavaScript
// thread through `invoker` while `alive` is true.
void install(facebook::jsi::Runtime &rt, std::shared_ptr<facebook::react::CallInvoker> invoker,
             std::shared_ptr<std::atomic<bool>> alive);

} // namespace morflow
