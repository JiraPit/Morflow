import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

/**
 * The native side of react-native-morflow, a C++ module. `install()` adds the
 * engine's functions to the JavaScript runtime as `global.__morflow`, so
 * pipelines can take and return ArrayBuffers without copying them through the
 * bridge. Not part of the package's API.
 */
export interface Spec extends TurboModule {
  install(): boolean;
}

export default TurboModuleRegistry.getEnforcing<Spec>('Morflow');
