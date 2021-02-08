use crate::rayt::*;

struct SimpleScene {}

impl SimpleScene {
    fn hit_sphere(&self, center: Point3, radius: f64, ray: &Ray) -> f64 {
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

    fn background(&self, d: Vec3) -> Color {
        let t = 0.5 * (d.normalize().y() + 1.0);
        Color::one().lerp(Color::new(0.5, 0.7, 1.0), t)
    }
}

impl Scene for SimpleScene {
    fn camera(&self) -> Camera {
        Camera::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-2.0, -1.0, -1.0),
        )
    }

    fn trace(&self, ray: Ray) -> Color {
        let c = Point3::new(0.0, 0.0, -1.0);
        let t = self.hit_sphere(c, 0.5, &ray);
        if t > 0.0 {
            let n = (ray.at(t) - c).normalize();
            return 0.5 * (n + Vec3::one());
        }
        self.background(ray.direction)
    }
}

pub fn run() {
    render(SimpleScene {});
}
