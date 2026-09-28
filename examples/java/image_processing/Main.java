import org.morflow.Morflow;
import org.morflow.MorflowTensor;
import org.morflow.Pipeline;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        // 1. Load pipeline and input image
        Pipeline pipeline = Morflow.load("image_pipeline.morf");
        BufferedImage inputImage = ImageIO.read(new File("input.png"));
        int inWidth = inputImage.getWidth();
        int inHeight = inputImage.getHeight();

        System.out.println("Loaded pipeline: " + pipeline);
        System.out.println("Input image: " + inWidth + "x" + inHeight + ", RGB");

        // Convert BufferedImage to HWC uint8 byte array [H, W, 3]
        byte[] rgbBytes = new byte[inHeight * inWidth * 3];
        int idx = 0;
        for (int y = 0; y < inHeight; y++) {
            for (int x = 0; x < inWidth; x++) {
                int rgb = inputImage.getRGB(x, y);
                rgbBytes[idx++] = (byte) ((rgb >> 16) & 0xFF); // R
                rgbBytes[idx++] = (byte) ((rgb >> 8) & 0xFF);  // G
                rgbBytes[idx++] = (byte) (rgb & 0xFF);         // B
            }
        }

        MorflowTensor inputTensor = MorflowTensor.fromByteArray(rgbBytes, new int[]{inHeight, inWidth, 3});

        // 2. Execute Morflow pipeline
        MorflowTensor outputTensor = pipeline.run(inputTensor);
        int[] outShape = outputTensor.getShape();
        int outHeight = outShape[0];
        int outWidth = outShape[1];
        int outChannels = outShape.length > 2 ? outShape[2] : 1;

        System.out.println("Output tensor: " + Arrays.toString(outShape) + ", " + outputTensor.getDtype());

        // 3. Save output image (RGBA) to disk
        byte[] outBytes = outputTensor.toByteArray();
        BufferedImage outImage = new BufferedImage(outWidth, outHeight, BufferedImage.TYPE_INT_ARGB);
        idx = 0;
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                int r = outBytes[idx++] & 0xFF;
                int g = outBytes[idx++] & 0xFF;
                int b = outBytes[idx++] & 0xFF;
                int a = outChannels == 4 ? (outBytes[idx++] & 0xFF) : 255;
                int argb = (a << 24) | (r << 16) | (g << 8) | b;
                outImage.setRGB(x, y, argb);
            }
        }

        ImageIO.write(outImage, "png", new File("output.png"));
        System.out.println("Successfully processed image with Java host and saved to output.png");
    }
}
