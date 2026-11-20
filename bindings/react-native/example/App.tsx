/**
 * Runs Morflow pipelines bundled with the app and shows the results. CI reads
 * the "MORFLOW" lines from the log to check each one.
 */
import React, { useEffect, useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text } from 'react-native';
import * as morflow from 'react-native-morflow';

import abs from './pipelines/abs.morf';
import boost from './pipelines/boost.morf';

type Check = { name: string; ok: boolean; detail: string };

/**
 * Logs a line for CI. iOS release builds drop every React Native log below
 * error level from the system log, so this uses console.error to be seen there.
 */
const report = (line: string) => console.error(line);

async function checks(): Promise<Check[]> {
  const results: Check[] = [];
  const check = async (name: string, body: () => Promise<string>) => {
    try {
      results.push({ name, ok: true, detail: await body() });
    } catch (error) {
      results.push({ name, ok: false, detail: String(error) });
    }
  };

  await check('abs on a worker thread', async () => {
    const pipeline = morflow.fromStr(abs);
    const out = await pipeline.run(new Float32Array([-1.5, 2, -3]));
    const values = [...out.toFloat32Array()];
    if (values.join(' ') !== '1.5 2 3') throw new Error(`got ${values.join(' ')}`);
    return `abs: ${values.join(' ')}`;
  });

  await check('abs on the JavaScript thread', async () => {
    const pipeline = morflow.fromStr(abs);
    const out = pipeline.runSync({ data: new Float64Array([-4, 5]), shape: [2] });
    const values = [...out.toFloat64Array()];
    pipeline.close();
    if (out.dtype !== 'f64' || values.join(' ') !== '4 5') throw new Error(`got ${out.dtype} ${values.join(' ')}`);
    return `${out.dtype}: ${values.join(' ')}`;
  });

  await check('audio gain to WAV', async () => {
    const pipeline = morflow.fromStr(boost);
    const samples = new Float32Array(2205).map((_, i) => 0.25 * Math.sin(i / 10));
    const wav = await pipeline.run({ data: samples, shape: [1, samples.length], payloadType: 'audio', sampleRate: 22050 });
    const bytes = wav.toBuffer();
    const header = String.fromCharCode(...bytes.subarray(0, 4));
    if (header !== 'RIFF') throw new Error(`not a WAV file (${header})`);
    return `${bytes.length}-byte WAV`;
  });

  await check('errors reach JavaScript', async () => {
    try {
      morflow.fromStr('accept Tensor $t\n$t >> no_such_action >> emit');
    } catch (error) {
      return String(error).slice(0, 60);
    }
    throw new Error('compiling an unknown action did not fail');
  });

  return results;
}

export default function App() {
  const [results, setResults] = useState<Check[]>([]);

  useEffect(() => {
    checks().then((done) => {
      for (const r of done) report(`MORFLOW ${r.ok ? 'PASS' : 'FAIL'} ${r.name}: ${r.detail}`);
      report(`MORFLOW DONE ${done.every((r) => r.ok) ? 'PASS' : 'FAIL'}`);
      setResults(done);
    });
  }, []);

  return (
    <SafeAreaView style={styles.screen}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Morflow {morflow.engineVersion}</Text>
        {results.length === 0 && <Text>Running pipelines…</Text>}
        {results.map((r) => (
          <Text key={r.name} style={r.ok ? styles.pass : styles.fail}>
            {r.ok ? '✓' : '✗'} {r.name}: {r.detail}
          </Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1 },
  content: { padding: 24, gap: 12 },
  title: { fontSize: 22, fontWeight: '600' },
  pass: { color: '#1a7f37' },
  fail: { color: '#cf222e' },
});
