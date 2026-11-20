// The TurboModule React Native finds as "Morflow". Its only method installs
// the engine's functions in the JavaScript runtime (see MorflowHost.h).
#pragma once

#include <RNMorflowSpecJSI.h>

#include <atomic>
#include <memory>

namespace facebook::react {

class MorflowModule : public NativeMorflowCxxSpec<MorflowModule> {
 public:
  explicit MorflowModule(std::shared_ptr<CallInvoker> jsInvoker);
  ~MorflowModule() override;

  bool install(jsi::Runtime &rt);

 private:
  // False once the runtime this module belongs to is going away, so runs that
  // finish later are dropped instead of touching it.
  std::shared_ptr<std::atomic<bool>> alive_;
};

} // namespace facebook::react
