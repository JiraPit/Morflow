// `import boost from './pipelines/boost.morf'` gives the pipeline's source
// text, through the Metro helper in react-native-morflow/metro.
declare module '*.morf' {
  const source: string;
  export default source;
}
