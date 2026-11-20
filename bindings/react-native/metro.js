// Lets an app import its .morf pipelines as text:
//
//   // metro.config.js
//   const { withMorflow } = require('react-native-morflow/metro');
//   module.exports = withMorflow(mergeConfig(getDefaultConfig(__dirname), {}));
//
//   // in the app
//   import boost from './pipelines/boost.morf';
//   const pipeline = morflow.fromStr(boost);
'use strict';

/** `config` that also bundles .morf files, each as a module exporting its text. */
function withMorflow(config) {
  const transformer = config.transformer || {};
  const resolver = config.resolver || {};
  // .morf files go through morf-transformer.js, which hands everything else
  // to the transformer the config already had (React Native's or Expo's).
  if (transformer.babelTransformerPath && transformer.babelTransformerPath !== require.resolve('./morf-transformer')) {
    process.env.MORFLOW_UPSTREAM_TRANSFORMER = transformer.babelTransformerPath;
  }
  const sourceExts = resolver.sourceExts || ['js', 'jsx', 'json', 'ts', 'tsx'];
  return {
    ...config,
    resolver: { ...resolver, sourceExts: sourceExts.includes('morf') ? sourceExts : [...sourceExts, 'morf'] },
    transformer: { ...transformer, babelTransformerPath: require.resolve('./morf-transformer') },
  };
}

module.exports = { withMorflow };
