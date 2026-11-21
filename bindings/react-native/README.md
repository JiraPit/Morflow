# Morflow for React Native

Process images, audio and tensors in your React Native app with Morflow pipelines. You write a pipeline once in a `.morf` file, and it runs inside your app on Android and iOS. It works offline, and your users don't need to install anything.

## Get started

**1. Add Morflow to your app**

```sh
npm install react-native-morflow
```

Then let Metro import `.morf` files by wrapping your `metro.config.js`:

```js
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const { withMorflow } = require('react-native-morflow/metro');

module.exports = withMorflow(mergeConfig(getDefaultConfig(__dirname), {}));
```

Using Expo? Add `"plugins": ["react-native-morflow"]` to `app.json` too, and use a development build: Expo Go can't load it.

**2. Write a pipeline**

Save it anywhere in your app's folder, such as `pipelines/boost.morf`. This one doubles the volume of a recording and returns it as a WAV file:

```
import audio_basics/latest

accept Audio $audio
$audio >> gain(linear=2.0) >> to_wav >> emit
```

**3. Get your app ready**

Install the Morflow CLI from [morflow.org](https://morflow.org/doc#cli-installation). Then run this in your app's folder, listing every pipeline your app uses:

```sh
morflow prep --prod-target react-native pipelines/boost.morf

# Or every pipeline in a folder
morflow prep --prod-target react-native pipelines
```

It shows what it will add to your `package.json` and asks before changing anything. Press Enter to accept.

**4. Run the pipeline**

```ts
import * as morflow from 'react-native-morflow';
import boost from './pipelines/boost.morf';

const pipeline = morflow.fromStr(boost);

export async function boostRecording(samples: Float32Array): Promise<Uint8Array> {
  const wav = await pipeline.run({ data: samples, shape: [1, samples.length], payloadType: 'audio', sampleRate: 44100 });
  return wav.toBuffer();
}
```

**5. Build as usual**

`npx react-native run-android` and `run-ios` (or `npx expo run:android` and `run:ios`) include everything your pipelines need. When you add or change a pipeline, run step 3 again, and `pod install` before building for iOS.

## Learn more

- [The React Native guide](https://morflow.org/doc#exec-react-native): every step in detail, inputs and outputs, and how it works
- [Writing pipelines](https://morflow.org/doc#syntax)
- [Data types: tensors, images, audio and more](https://morflow.org/doc#types)
- [Questions and issues on GitHub](https://github.com/JiraPit/Morflow/issues)
