use crate::rayt::*;
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::{fs, path::Path};

const IMAGE_WIDTH: u32 = 200;
const IMAGE_HEIGHT: u32 = 100;
const OUTPUT_FILENAME: &str = "render.png";
const BACKUP_FILENAME: &str = "render_bak.png";

fn backup() {
    let output_path = Path::new(OUTPUT_FILENAME);
    if output_path.exists() {
        println!("backup {:?} -> {:?}", OUTPUT_FILENAME, BACKUP_FILENAME);
        // replacing the original file if to already exists
        fs::rename(OUTPUT_FILENAME, BACKUP_FILENAME).unwrap();
    }
}

fn hit_sphere(center: Point3, radius: f64, ray: &Ray) -> f64 {
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * ray.direction.dot(oc);
    let c = oc.dot(oc) - radius.powi(2);
    let d = b * b - 4.0 * a * c;
    if d < 0.0 {
        -1.0
    } else {
        return (-b - d.sqrt()) / (2.0 * a);
    }
}

fn color(ray: Ray) -> Color {
    let c = Point3::new(0.0, 0.0, -1.0);
    let t = hit_sphere(c, 0.5, &ray);
    if t > 0.0 {
        let n = (ray.at(t) - c).normalize();
        return 0.5 * (n + Vec3::one());
    }
    let d = ray.direction.normalize();
    let t = 0.5 * (d.y() + 1.0);
    Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
}

pub fn run() {
    backup();

    let camera = Camera::new(
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-2.0, -1.0, -1.0),
    );
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);
    img.enumerate_pixels_mut()
        .collect::<Vec<(u32, u32, &mut Rgb<u8>)>>()
        .par_iter_mut()
        .for_each(|(x, y, pixel)| {
            let u = *x as f64 / (IMAGE_WIDTH - 1) as f64;
            let v = (IMAGE_HEIGHT - *y - 1) as f64 / (IMAGE_HEIGHT - 1) as f64;
            let ray = camera.ray(u, v);
            let rgb = color(ray).to_rgb();
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        });
    img.save(OUTPUT_FILENAME).unwrap();
    draw_in_window(BACKUP_FILENAME, img).unwrap();
}
