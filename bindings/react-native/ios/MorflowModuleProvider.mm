#import "MorflowModuleProvider.h"

#import "MorflowModule.h"

@implementation MorflowModuleProvider

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
  return std::make_shared<facebook::react::MorflowModule>(params.jsInvoker);
}

@end
