#import <Foundation/Foundation.h>
#import <ReactCommon/RCTTurboModule.h>

/// Gives React Native the C++ module (cpp/MorflowModule.h) on iOS; named in
/// package.json's codegenConfig.ios.modulesProvider.
@interface MorflowModuleProvider : NSObject <RCTModuleProvider>
@end
