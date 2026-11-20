// The Expo config plugin: `"plugins": ["react-native-morflow"]` in app.json.
// Morflow's native libraries are published for 64-bit Android (arm64-v8a and
// x86_64), so the app builds for those ABIs only. iOS needs no changes.
'use strict';

const { withGradleProperties } = require('expo/config-plugins');

const ABIS = 'arm64-v8a,x86_64';

function withMorflow(config) {
  return withGradleProperties(config, (gradle) => {
    gradle.modResults = gradle.modResults.filter(
      (item) => !(item.type === 'property' && item.key === 'reactNativeArchitectures')
    );
    gradle.modResults.push({ type: 'property', key: 'reactNativeArchitectures', value: ABIS });
    return gradle;
  });
}

module.exports = withMorflow;
