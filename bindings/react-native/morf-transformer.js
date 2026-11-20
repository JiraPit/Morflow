// The Metro transformer withMorflow (metro.js) installs: a .morf file becomes
// a module that exports its text; every other file goes to the transformer the
// app already used.
'use strict';

function upstream() {
  const name = process.env.MORFLOW_UPSTREAM_TRANSFORMER || '@react-native/metro-babel-transformer';
  return require(require.resolve(name, { paths: [process.cwd(), __dirname] }));
}

const base = upstream();

module.exports = {
  ...base,
  transform(props) {
    if (props.filename.endsWith('.morf')) {
      return base.transform({ ...props, src: `module.exports = ${JSON.stringify(props.src)};\n` });
    }
    return base.transform(props);
  },
};
