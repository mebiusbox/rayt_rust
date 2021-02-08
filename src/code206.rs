use crate::rayt::*;

// ScatterInfo

struct ScatterInfo {
    ray: Ray,
    albedo: Color,
}

impl ScatterInfo {
    fn new(ray: Ray, albedo: Color) -> Self {
        Self { ray, albedo }
    }
}

// Textures

trait Texture: Sync + Send {
    fn value(&self, u: f64, v: f64, p: Point3) -> Color;
}

struct ColorTexture {
    color: Color,
}

impl ColorTexture {
    const fn new(color: Color) -> Self {
        Self { color }
    }
}

impl Texture for ColorTexture {
    fn value(&self, _u: f64, _v: f64, _p: Point3) -> Color {
        self.color
    }
}

struct CheckerTexture {
    odd: Box<dyn Texture>,
    even: Box<dyn Texture>,
    freq: f64,
}

impl CheckerTexture {
    fn new(odd: Box<dyn Texture>, even: Box<dyn Texture>, freq: f64) -> Self {
        Self { odd, even, freq }
    }
}

impl Texture for CheckerTexture {
    fn value(&self, u: f64, v: f64, p: Point3) -> Color {
        let sines = p.iter().fold(1.0, |acc, x| acc * (x * self.freq).sin());
        if sines < 0.0 {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}

struct ImageTexture {
    pixels: Vec<Color>,
    width: usize,
    height: usize,
}

impl ImageTexture {
    fn new(path: &str) -> Self {
        let rgbimg = image::open(path).unwrap().to_rgb8();
        let (w, h) = rgbimg.dimensions();
        let mut image = vec![Color::zero(); (w * h) as usize];
        for (i, (_, _, pixel)) in image.iter_mut().zip(rgbimg.enumerate_pixels()) {
            *i = Color::from_rgb(pixel[0], pixel[1], pixel[2]);
        }
        Self { pixels: image, width: w as usize, height: h as usize }
    }

    fn sample(&self, u: i64, v: i64) -> Color {
        let tu = if u < 0 { 0 } else if u as usize >= self.width { self.width - 1 } else { u as usize };
        let tv = if v < 0 { 0 } else if v as usize >= self.height { self.height - 1 } else { v as usize };
        self.pixels[tu + self.width * tv]
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, _p: Point3) -> Color {
        let x = (u * self.width as f64) as i64;
        let y = ((1.0 - v) * self.height as f64) as i64;
        self.sample(x, y)
    }
}

// Materials

trait Material: Sync + Send {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo>;
    fn emitted(&self, _ray: &Ray, _hit: &HitInfo) -> Color { Color::zero() }
}

struct Lambertian {
    albedo: Box<dyn Texture>,
}

impl Lambertian {
    fn new(albedo: Box<dyn Texture>) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let target = hit.p + hit.n + Vec3::random_in_unit_sphere();
        let albedo = self.albedo.value(hit.u, hit.v, hit.p);
        Some(ScatterInfo::new(Ray::new(hit.p, target - hit.p), albedo))
    }
}

struct Metal {
    albedo: Box<dyn Texture>,
    fuzz: f64,
}

impl Metal {
    fn new(albedo: Box<dyn Texture>, fuzz: f64) -> Self {
        Self { albedo, fuzz }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let mut reflected = ray.direction.normalize().reflect(hit.n);
        reflected = reflected + self.fuzz * Vec3::random_in_unit_sphere();
        if reflected.dot(hit.n) > 0.0 {
            let albedo = self.albedo.value(hit.u, hit.v, hit.p);
            Some(ScatterInfo::new(Ray::new(hit.p, reflected), albedo))
        } else {
            None
        }
    }
}

struct Dielectric {
    ri: f64,
}

impl Dielectric {
    const fn new(ri: f64) -> Self {
        Self { ri }
    }

    /// Compute schlick fresnel approximation
    fn schlick(cosine: f64, ri: f64) -> f64 {
        let r0 = ((1.0 - ri) / (1.0 + ri)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitInfo) -> Option<ScatterInfo> {
        let reflected = ray.direction.reflect(hit.n);
        let (outward_normal, ni_over_nt, cosine) = {
            let dot = ray.direction.dot(hit.n);
            if dot > 0.0 {
                (-hit.n, self.ri, self.ri * dot / ray.direction.length())
            } else {
                (hit.n, self.ri.recip(), -dot / ray.direction.length())
            }
        };

        if let Some(refracted) = (-ray.direction).refract(outward_normal, ni_over_nt) {
            if Vec3::random_full().x() > Self::schlick(cosine, self.ri) {
                return Some(ScatterInfo::new(Ray::new(hit.p, refracted), Color::one()));
            }
        }

        Some(ScatterInfo::new(Ray::new(hit.p, reflected), Color::one()))
    }
}

struct DiffuseLight {
    emit: Box<dyn Texture>,
}

impl DiffuseLight {
    fn new(emit: Box<dyn Texture>) -> Self {
        Self { emit }
    }
}

impl Material for DiffuseLight {
    fn scatter(&self, _ray: &Ray, _hit: &HitInfo) -> Option<ScatterInfo> {
        None
    }

    fn emitted(&self, _ray: &Ray, hit: &HitInfo) -> Color {
        self.emit.value(hit.u, hit.v, hit.p)
    }
}

// HitInfo

struct HitInfo {
    t: f64,
    p: Point3,
    n: Vec3,
    m: Arc<dyn Material>,
    u: f64,
    v: f64,
}

impl HitInfo {
    fn new(t: f64, p: Point3, n: Vec3, m: Arc<dyn Material>, u: f64, v: f64) -> Self {
        Self { t, p, n, m, u, v }
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
    material: Arc<dyn Material>,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Arc<dyn Material>) -> Self {
        Self { center, radius, material }
    }

    fn uv(p: Point3) -> (f64, f64) {
        let phi = p.z().atan2(p.x());
        let theta = p.y().asin();
        (1.0 - (phi + PI) / PI2, (theta + PI / 2.0) * FRAC_1_PI)
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
                let n = (p - self.center) / self.radius;
                let (u, v) = Self::uv(n);
                return Some(HitInfo::new(temp, p, n, Arc::clone(&self.material), u, v));
            }
            let temp = (-b + root) / (2.0 * a);
            if t0 < temp && temp < t1 {
                let p = ray.at(temp);
                let n = (p - self.center) / self.radius;
                let (u, v) = Self::uv(n);
                return Some(HitInfo::new(temp, p, n, Arc::clone(&self.material), u, v));
            }
        }

        None
    }
}

enum RectAxisType {
    XY,
    XZ,
    YZ,
}

struct Rect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    k: f64,
    axis: RectAxisType,
    material: Arc<dyn Material>,
}

impl Rect {
    fn new(x0: f64, x1: f64, y0: f64, y1: f64, k: f64, axis: RectAxisType, material: Arc<dyn Material>) -> Self {
        Self { x0, x1, y0, y1, k, axis, material }
    }
}

impl Shape for Rect {
    fn hit(&self, ray: &Ray, t0: f64, t1: f64) -> Option<HitInfo> {
        let mut origin = ray.origin;
        let mut direction = ray.direction;
        let mut axis = Vec3::zaxis();
        match self.axis {
            RectAxisType::XY => {}
            RectAxisType::XZ => {
                origin = Point3::new(origin.x(), origin.z(), origin.y());
                direction = Vec3::new(direction.x(), direction.z(), direction.y());
                axis = Vec3::yaxis();
            }
            RectAxisType::YZ => {
                origin = Point3::new(origin.y(), origin.z(), origin.x());
                direction = Vec3::new(direction.y(), direction.z(), direction.x());
                axis = Vec3::xaxis();
            }
        }

        let t = (self.k - origin.z()) / direction.z();
        if t < t0 || t > t1 {
            return None;
        }

        let x = origin.x() + t * direction.x();
        let y = origin.y() + t * direction.y();
        if x < self.x0 || x > self.x1 || y < self.y0 || y > self.y1 {
            return None;
        }

        Some(HitInfo::new(
            t,
            ray.at(t),
            axis,
            Arc::clone(&self.material),
            (x - self.x0) / (self.x1 - self.x0),
            (y - self.y0) / (self.y1 - self.y0),
        ))
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

// Shape Builder

struct ShapeBuilder {
    texture: Option<Box<dyn Texture>>,
    material: Option<Arc<dyn Material>>,
    shape: Option<Box<dyn Shape>>,
}

impl ShapeBuilder {
    fn new() -> Self {
        Self { texture: None, material: None, shape: None }
    }

    // textures

    fn color_texture(mut self, color: Color) -> Self {
        self.texture = Some(Box::new(ColorTexture::new(color)));
        self
    }

    fn checker_texture(mut self, odd_color: Color, even_color: Color, freq: f64) -> Self {
        self.texture = Some(Box::new(CheckerTexture::new(
            Box::new(ColorTexture::new(odd_color)),
            Box::new(ColorTexture::new(even_color)),
            freq,
        )));
        self
    }

    fn image_texture(mut self, path: &str) -> Self {
        self.texture = Some(Box::new(ImageTexture::new(path)));
        self
    }

    // materials

    fn lambertian(mut self) -> Self {
        self.material = Some(Arc::new(Lambertian::new(self.texture.unwrap())));
        self.texture = None;
        self
    }

    fn metal(mut self, fuzz: f64) -> Self {
        self.material = Some(Arc::new(Metal::new(self.texture.unwrap(), fuzz)));
        self.texture = None;
        self
    }

    fn dielectric(mut self, ri: f64) -> Self {
        self.material = Some(Arc::new(Dielectric::new(ri)));
        self
    }

    fn diffuse_light(mut self) -> Self {
        self.material = Some(Arc::new(DiffuseLight::new(self.texture.unwrap())));
        self.texture = None;
        self
    }

    // shapes

    fn sphere(mut self, center: Point3, radius: f64) -> Self {
        self.shape = Some(Box::new(Sphere::new(center, radius, self.material.unwrap())));
        self.material = None;
        self
    }

    fn rect_xy(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::XY, self.material.unwrap())));
        self.material = None;
        self
    }

    fn rect_xz(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::XZ, self.material.unwrap())));
        self.material = None;
        self
    }

    fn rect_yz(mut self, x0: f64, x1: f64, y0: f64, y1: f64, k: f64) -> Self {
        self.shape = Some(Box::new(Rect::new(x0, x1, y0, y1, k, RectAxisType::YZ, self.material.unwrap())));
        self.material = None;
        self
    }

    // build

    fn build(self) -> Box<dyn Shape> {
        self.shape.unwrap()
    }
}

// Scene

struct CornelBoxScene {
    world: ShapeList,
}

impl CornelBoxScene {
    fn new() -> Self {
        let mut world = ShapeList::new();

        let red = Color::new(0.64, 0.05, 0.05);
        let white = Color::full(0.73);
        let green = Color::new(0.12, 0.45, 0.15);
    
        world.push(ShapeBuilder::new()
            .color_texture(green)
            .lambertian()
            .rect_yz(0.0, 555.0, 0.0, 555.0, 555.0)
            .build());
        world.push(ShapeBuilder::new()
            .color_texture(red)
            .lambertian()
            .rect_yz(0.0, 555.0, 0.0, 555.0, 0.0)
            .build());
        world.push(ShapeBuilder::new()
            .color_texture(Color::full(15.0))
            .diffuse_light()
            .rect_xz(213.0, 343.0, 227.0, 332.0, 554.0)
            .build());
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xz(0.0, 555.0, 0.0, 555.0, 555.0)
            .build());
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xz(0.0, 555.0, 0.0, 555.0, 0.0)
            .build());
        world.push(ShapeBuilder::new()
            .color_texture(white)
            .lambertian()
            .rect_xy(0.0, 555.0, 0.0, 555.0, 555.0)
            .build());
        
        Self { world }
    }

    fn background(&self, _: Vec3) -> Color {
        Color::full(0.0)
    }
}

impl SceneWithDepth for CornelBoxScene {
    fn camera(&self) -> Camera {
        Camera::from_lookat(
            Vec3::new(278.0, 278.0, -800.0),
            Vec3::new(278.0, 278.0, 0.0),
            Vec3::yaxis(),
            40.0,
            self.aspect(),
        )
    }

    fn trace(&self, ray: Ray, depth: usize) -> Color {
        let hit_info = self.world.hit(&ray, 0.001, f64::MAX);
        if let Some(hit) = hit_info {
            let emitted = hit.m.emitted(&ray, &hit);
            let scatter_info = if depth > 0 { hit.m.scatter(&ray, &hit) } else { None };
            if let Some(scatter) = scatter_info {
                emitted + scatter.albedo * self.trace(scatter.ray, depth - 1)
            } else {
                emitted
            }
        } else {
            self.background(ray.direction)
        }
    }

    fn width(&self) -> u32 { 200 }
    fn height(&self) -> u32 { 200 }
}

pub fn run() {
    render_aa_with_depth(CornelBoxScene::new());
}
