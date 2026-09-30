import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import morflow from '../../../bindings/js/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Zero-dependency pure Node.js PNG decoder (RGBA / RGB)
function decodePng(buffer) {
  let offset = 8; // Skip PNG header
  let width = 0, height = 0, colorType = 0;
  const idatChunks = [];

  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString('ascii', offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);

    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      colorType = data[9]; // 2 = RGB, 6 = RGBA
    } else if (type === 'IDAT') {
      idatChunks.push(data);
    } else if (type === 'IEND') {
      break;
    }
    offset += 12 + length;
  }

  const compressed = Buffer.concat(idatChunks);
  const decompressed = zlib.inflateSync(compressed);

  const channels = colorType === 6 ? 4 : 3;
  const stride = width * channels;
  const rawBytes = Buffer.alloc(width * height * channels);

  let srcOffset = 0;
  let dstOffset = 0;

  for (let y = 0; y < height; y++) {
    const filterType = decompressed[srcOffset++];
    const prevRowOffset = dstOffset - stride;

    for (let x = 0; x < stride; x++) {
      const raw = decompressed[srcOffset++];
      const a = x >= channels ? rawBytes[dstOffset - channels] : 0;
      const b = y > 0 ? rawBytes[prevRowOffset + x] : 0;
      const c = (y > 0 && x >= channels) ? rawBytes[prevRowOffset + x - channels] : 0;

      let val = raw;
      if (filterType === 1) val = (raw + a) & 0xff; // Sub
      else if (filterType === 2) val = (raw + b) & 0xff; // Up
      else if (filterType === 3) val = (raw + Math.floor((a + b) / 2)) & 0xff; // Average
      else if (filterType === 4) { // Paeth
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        const pr = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
        val = (raw + pr) & 0xff;
      }
      rawBytes[dstOffset++] = val;
    }
  }

  // Convert to RGB if RGBA
  let rgbBytes = rawBytes;
  if (channels === 4) {
    rgbBytes = Buffer.alloc(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      rgbBytes[i * 3] = rawBytes[i * 4];
      rgbBytes[i * 3 + 1] = rawBytes[i * 4 + 1];
      rgbBytes[i * 3 + 2] = rawBytes[i * 4 + 2];
    }
  }

  return { width, height, data: rgbBytes };
}

// Zero-dependency pure Node.js PNG encoder for RGBA
function encodePng(rgbaBytes, width, height) {
  const stride = width * 4;
  const rawData = Buffer.alloc(height * (stride + 1));

  let srcOffset = 0;
  let dstOffset = 0;
  for (let y = 0; y < height; y++) {
    rawData[dstOffset++] = 0; // Filter 0 (None)
    rgbaBytes.copy(rawData, dstOffset, srcOffset, srcOffset + stride);
    dstOffset += stride;
    srcOffset += stride;
  }

  const compressed = zlib.deflateSync(rawData);

  // CRC32 table
  const crcTable = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crcTable[n] = c >>> 0;
  }

  function crc32(buf) {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) {
      c = (c >>> 8) ^ crcTable[(c ^ buf[i]) & 0xff];
    }
    return (c ^ 0xffffffff) >>> 0;
  }

  function makeChunk(type, data) {
    const len = data.length;
    const buf = Buffer.alloc(12 + len);
    buf.writeUInt32BE(len, 0);
    buf.write(type, 4);
    data.copy(buf, 8);
    const crc = crc32(buf.subarray(4, 8 + len));
    buf.writeUInt32BE(crc, 8 + len);
    return buf;
  }

  const header = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // RGBA color type
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const ihdrChunk = makeChunk('IHDR', ihdr);
  const idatChunk = makeChunk('IDAT', compressed);
  const iendChunk = makeChunk('IEND', Buffer.alloc(0));

  return Buffer.concat([header, ihdrChunk, idatChunk, iendChunk]);
}

async function main() {
  process.chdir(__dirname);

  // 1. Load pipeline and decode input PNG
  const pipeline = morflow.load('image_pipeline.morf');
  const inputBuffer = fs.readFileSync('input.png');
  const img = decodePng(inputBuffer);

  console.log('Loaded pipeline: image_pipeline.morf');
  console.log(`Input image: ${img.width}x${img.height}, RGB`);

  // 2. Prepare Morflow TensorInput [H, W, 3]
  const tensorInput = {
    data: img.data,
    shape: [img.height, img.width, 3],
    dtype: 'u8',
    payloadType: 'image',
    colorSpace: 'rgb'
  };

  // 3. Execute Morflow image pipeline asynchronously
  const outputTensor = await pipeline.run(tensorInput);
  const outShape = outputTensor.shape; // [H, W, 4] RGBA
  const outHeight = outShape[0];
  const outWidth = outShape[1];
  const outData = outputTensor.toBuffer();

  console.log(`Output image: ${outWidth}x${outHeight}, RGBA`);

  // 4. Encode and save to output.png
  const encoded = encodePng(outData, outWidth, outHeight);
  fs.writeFileSync('output.png', encoded);
  console.log('Successfully processed image with JavaScript Node.js host and saved to output.png');
}

main().catch(console.error);
