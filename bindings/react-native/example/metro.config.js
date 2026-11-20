const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const { withMorflow } = require('react-native-morflow/metro');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * withMorflow lets the app import its .morf pipelines as text.
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {};

module.exports = withMorflow(mergeConfig(getDefaultConfig(__dirname), config));
