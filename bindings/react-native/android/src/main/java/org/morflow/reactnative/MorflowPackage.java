package org.morflow.reactnative;

import androidx.annotation.Nullable;
import com.facebook.react.BaseReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.model.ReactModuleInfoProvider;
import java.util.HashMap;

/**
 * Lets React Native's autolinking find this library. The module itself is
 * C++ (cpp/MorflowModule.h), so there is no Java module to provide.
 */
public class MorflowPackage extends BaseReactPackage {
  @Override
  public @Nullable NativeModule getModule(String name, ReactApplicationContext reactContext) {
    return null;
  }

  @Override
  public ReactModuleInfoProvider getReactModuleInfoProvider() {
    return HashMap::new;
  }
}
