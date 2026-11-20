// How React Native's autolinking builds this library: on Android, the C++
// module goes into the app's native library from android/CMakeLists.txt.
module.exports = {
  dependency: {
    platforms: {
      android: {
        cxxModuleCMakeListsModuleName: 'react_native_morflow',
        cxxModuleCMakeListsPath: 'CMakeLists.txt',
        cxxModuleHeaderName: 'MorflowModule',
      },
    },
  },
};
