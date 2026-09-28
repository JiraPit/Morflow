use core_types::{ColorSpace, Image, Payload};
use morflow::Morflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR"))?;

    // 1. Load pipeline and decode input image
    let mut pipeline = Morflow::load("image_pipeline.morf")?;
    let img = image::open("input.png")?.to_rgb8();

    // 2. Wrap image buffer into a Morflow Image payload
    let input = Image::from_u8_hwc(
        &img,
        img.width() as usize,
        img.height() as usize,
        ColorSpace::Rgb,
    )
    .map_err(|e| e.to_string())?;

    // 3. Execute Morflow pipeline
    let outputs = pipeline.run(Payload::Image(input))?;

    // 4. Extract emitted payload and save output PNG
    if let Payload::Image(out) = outputs.into_single()? {
        let bytes = out.to_contiguous_bytes();
        image::save_buffer(
            "output.png",
            bytes.as_slice(),
            out.width() as u32,
            out.height() as u32,
            image::ColorType::Rgba8,
        )?;
        println!("Successfully processed image and saved to output.png");
    }

    Ok(())
}
