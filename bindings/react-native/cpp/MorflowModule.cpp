#include "MorflowModule.h"

#include "MorflowHost.h"

namespace facebook::react {

MorflowModule::MorflowModule(std::shared_ptr<CallInvoker> jsInvoker)
    : NativeMorflowCxxSpec(std::move(jsInvoker)), alive_(std::make_shared<std::atomic<bool>>(true)) {}

MorflowModule::~MorflowModule() {
  alive_->store(false);
}

bool MorflowModule::install(jsi::Runtime &rt) {
  morflow::install(rt, jsInvoker_, alive_);
  return true;
}

} // namespace facebook::react
