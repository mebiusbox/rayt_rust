use crate::rayt::*;

// HitInfo

struct HitInfo {
    t: f64,
    p: Point3,
    n: Vec3,
}

impl HitInfo {
    const fn new(t: f64, p: Point3, n: Vec3) -> Self {
        Self { t, p, n }
    }
}

// Shape

trait Shape: Sync {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo>;
}

// Sphere

struct Sphere {
    center: Point3,
    radius: f64,
}

impl Sphere {
    const fn new(center: Point3, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Shape for Sphere {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let oc = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2.0 * ray.direction.dot(oc);
        let c = oc.dot(oc) - self.radius.powi(2);
        let d = b * b - 4.0 * a * c;
        if d > 0.0 {
            let root = d.sqrt();
            let temp = (-b - root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                return Some(HitInfo::new(temp, p, (p - self.center) / self.radius));
            }
            let temp = (-b + root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                return Some(HitInfo::new(temp, p, (p - self.center) / self.radius));
            }
        }

        None
    }
}

// ShapeList

struct ShapeList {
    pub objects: Vec<Box<dyn Shape>>,
}

impl ShapeList {
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }

    pub fn push(&mut self, object: Box<dyn Shape>) {
        self.objects.push(object);
    }
}

impl Shape for ShapeList {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let mut hit_info: Option<HitInfo> = None;
        let mut closest_so_far = t1;
        for object in &self.objects {
            if let Some(info) = object.hit(ray, t0, closest_so_far) {
                closest_so_far = info.t;
                hit_info = Some(info);
            }
        }

        hit_info
    }
}

// Scene

struct SimpleScene {
    world: ShapeList,
}

impl SimpleScene {
    fn new() -> Self {
        let mut world = ShapeList::new();
        world.push(Box::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)));
        world.push(Box::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)));
        Self { world }
    }

    fn background(&self, d: &Vec3) -> Color {
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
        let hit_info = self.world.hit(&ray, 0.0, f64::MAX);
        if let Some(hit) = hit_info {
            let target = hit.p + hit.n + Vec3::random_in_unit_sphere();
            0.5 * self.trace(Ray::new(hit.p, target - hit.p))
        } else {
            self.background(&ray.direction)
        }
    }
}

pub fn run() {
    render_aa(SimpleScene::new());
}
