use minifb::{Key, KeyRepeat, Window, WindowOptions};
use image::RgbImage;

pub fn draw_in_window(backup_filename: &str, pixels: RgbImage) -> minifb::Result<()> {
    if cfg!(test) { return Ok(()) }
    let (image_width, image_height) = pixels.dimensions();
    let mut buffer: Vec<u32> = vec![0; (image_width * image_height) as usize];
    let mut window = Window::new(
        "ESC to exit",
        image_width as usize,
        image_height as usize,
        WindowOptions {
            topmost: true,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}", e));

    // Limit to max ~30 fps update here
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600 * 2)));

    for (i, (_, _, pixel)) in buffer.iter_mut().zip(pixels.enumerate_pixels()) {
        *i = u32::from_be_bytes([0, pixel[0], pixel[1], pixel[2]]);
    }

    window.update_with_buffer(&buffer, image_width as usize, image_height as usize)?;

    let mut backup_buffer: Option<Vec<u32>> = None;
    let mut show_backup = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if window.is_key_pressed(Key::D, KeyRepeat::No) {
            if backup_buffer.is_none() {
                if let Ok(img) = image::open(backup_filename) {
                    let backup_image = img.to_rgb8();
                    let (w, h) = backup_image.dimensions();
                    let mut buf = vec![0; (w * h) as usize];
                    for (i, (_, _, pixel)) in buf.iter_mut().zip(backup_image.enumerate_pixels()) {
                        *i = u32::from_be_bytes([0, pixel[0], pixel[1], pixel[2]]);
                    }

                    backup_buffer = Some(buf);
                }
            }

            show_backup = !show_backup;
        }

        let mut current_buffer = &buffer;
        if show_backup {
            if let Some(ref x) = backup_buffer {
                current_buffer = x;
            }
        }
        window.update_with_buffer(current_buffer, image_width as usize, image_height as usize)?;
    }

    Ok(())
}
