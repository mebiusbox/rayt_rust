use crate::rayt::*;
use rand::prelude::*;
use std::iter::FromIterator;

/// A 3-element array that is represented by double precision floating point components
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Float3([f64; 3]);

/// A 3-element color that is represented by r,g,b components
pub type Color = Float3;

/// A 3-element vector that is represented by x,y,z coordinates
pub type Vec3 = Float3;

/// A 3-element point that is represented by x,y,z coordinates
pub type Point3 = Float3;

impl Float3 {
    /// Construct from x,y,z elements
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    /// Construct 3-element array filled with zeros
    pub const fn zero() -> Self {
        Self([0.0; 3])
    }

    /// Construct 3-element array filled with ones
    pub const fn one() -> Self {
        Self([1.0; 3])
    }

    /// Construct 3-element array filled with value
    pub const fn full(value: f64) -> Self {
        Self([value; 3])
    }

    /// Compute the square root of the 3-element array per element
    pub fn sqrt(&self) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.sqrt()))
    }

    /// Tests all elements of the the 3-element array at or near zero
    pub fn near_zero(&self) -> bool {
        self.0.iter().all(|x| x.abs() < EPS)
    }

    /// Returns the clamped 3-element array. [0..1]
    pub fn saturate(&self) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.min(1.0).max(0.0)))
    }

    /// Returns 3-element array
    pub fn to_array(&self) -> [f64; 3] {
        self.0
    }

    /// Returns iterator of 3-element array
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }

    /// Returns iterator of 3-element array
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.0.iter_mut()
    }
}

/// Construct from iterator
impl FromIterator<f64> for Float3 {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut initer = iter.into_iter();
        Float3([
            initer.next().unwrap(),
            initer.next().unwrap(),
            initer.next().unwrap(),
        ])
    }
}

/// implements vector mathmatics
impl Float3 {
    /// Compute the dot product of two vectors
    pub fn dot(&self, rhs: Self) -> f64 {
        self.0.iter().zip(rhs.0.iter()).fold(0.0, |acc, (l,r)| acc + l*r)
    }

    /// Compute the cross product of two vectors
    pub fn cross(&self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }

    /// Compute the length of vector
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Compute the squared length of vector
    pub fn length_squared(&self) -> f64 {
        self.0.iter().fold(0.0, |acc, x| acc + x * x)
    }

    /// Returns normalized this vector
    pub fn normalize(&self) -> Self {
        *self / self.length()
    }

    /// Compute a reflect vector
    pub fn reflect(&self, normal: Self) -> Self {
        *self - 2.0 * self.dot(normal) * normal
    }

    /// Compute a refract vector
    pub fn refract(&self, normal: Self, ni_over_nt: f64) -> Option<Float3> {
        let uv = self.normalize();
        let dt = uv.dot(normal);
        let d = 1.0 - ni_over_nt.powi(2) * (1.0 - dt.powi(2));
        if d > 0.0 {
            Some(-ni_over_nt * (uv - normal * dt) - normal * d.sqrt())
        } else {
            None
        }
    }

    /// Compute linear interpolation between two vectors
    pub fn lerp(&self, v: Self, t: f64) -> Self {
        *self + (v - *self) * t
    }

    pub fn x(&self) -> f64 { self.0[0] }
    pub fn y(&self) -> f64 { self.0[1] }
    pub fn z(&self) -> f64 { self.0[2] }

    /// Returns a x-axis vector
    pub const fn xaxis() -> Self { Self::new(1.0, 0.0, 0.0) }
    /// Returns a y-axis vector
    pub const fn yaxis() -> Self { Self::new(0.0, 1.0, 0.0) }
    /// Returns a z-axis vector
    pub const fn zaxis() -> Self { Self::new(0.0, 0.0, 1.0) }
}

/// implements color utilities
impl Float3 {
    /// Construct from a hex slice, ex. b"ffffff"
    pub fn from_hex(hex: &[u8; 6]) -> Self {
        if let Ok(hex_str) = std::str::from_utf8(hex) {
            let r = u8::from_str_radix(&hex_str[0..2], 16).unwrap();
            let g = u8::from_str_radix(&hex_str[2..4], 16).unwrap();
            let b = u8::from_str_radix(&hex_str[4..6], 16).unwrap();
            Self::from_rgb(r, g, b)
        } else {
            panic!();
        }
    }

    /// Construct from r,g,b components. [0..255]
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0)
    }

    /// Returns the array that is represented by r,g,b components
    pub fn to_rgb(&self) -> [u8; 3] {
        [self.r(), self.g(), self.b()]
    }

    pub fn r(&self) -> u8 { (255.99 * self.0[0].min(1.0).max(0.0)) as u8 }
    pub fn g(&self) -> u8 { (255.99 * self.0[1].min(1.0).max(0.0)) as u8 }
    pub fn b(&self) -> u8 { (255.99 * self.0[2].min(1.0).max(0.0)) as u8 }

    /// Convert linear space to gamma space
    pub fn gamma(&self, factor: f64) -> Self {
        let recip = factor.recip();
        Self::from_iter(self.0.iter().map(|x| x.powf(recip)))
    }

    /// Convert gamma space to linear space
    pub fn degamma(&self, factor: f64) -> Self {
        Self::from_iter(self.0.iter().map(|x| x.powf(factor)))
    }
}

/// implements random utilities
impl Float3 {

    /// Construct 3-element array with a generated random values. [0,1)
    pub fn random() -> Self {
        Self::new(random::<f64>(), random::<f64>(), random::<f64>())
    }

    /// Construct 3-element array filled with a generated random values. [0,1)
    pub fn random_full() -> Self {
        Self::full(random::<f64>())
    }

    /// Construct 3-element array with a generated random values [min,max)
    pub fn random_limit(min: f64, max: f64) -> Self {
        Self::from_iter(Self::random().0.iter().map(|x| min + x * (max - min)))
    }

    /// Construct a random point in a unit sphere
    pub fn random_in_unit_sphere() -> Self {
        loop {
            let point = Self::random_limit(-1.0, 1.0);
            if point.length_squared() < 1.0 {
                return point;
            }
        }
    }

    /// Construct a random vector
    pub fn random_unit_vector() -> Self {
        Self::random_in_unit_sphere().normalize()
    }

    /// Construct a random point in a hemisphere
    pub fn random_in_hemisphere(normal: Self) -> Self {
        let in_unit_sphere = Self::random_in_unit_sphere();
        if in_unit_sphere.dot(normal) > 0.0 {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }

    /// Construct a random point in a unit disk
    pub fn random_in_unit_disk() -> Self {
        loop {
            let mut p = Self::random_limit(-1.0, 1.0);
            p.0[2] = 0.0;
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    /// Construct a random direction
    pub fn random_cosine_direction() -> Self {
        let Self([r1, r2, _]) = Self::random();
        let z = (1.0 - r2).sqrt();
        let (x, y) = (PI2 * r1).sin_cos();
        let r2sqrt = r2.sqrt();
        Self::new(x * r2sqrt, y * r2sqrt, z)
    }

    /// Construct a random vector to sphere
    pub fn random_to_sphere(radius: f64, distance_squared: f64) -> Self {
        let Self([rx, ry, _]) = Self::random();
        let rr = radius.powi(2).min(distance_squared);
        let cos_theta_max = (1.0 - rr * distance_squared.recip()).sqrt();
        let z = 1.0 - ry * (1.0 - cos_theta_max);
        let sqrtz = (1.0 - z.powi(2)).sqrt();
        let (x, y) = (PI2 * rx).sin_cos();
        Self::new(x * sqrtz, y * sqrtz, z)
    }
}

/// Unary negate
impl std::ops::Neg for Float3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_iter(self.0.iter().map(|x| -x))
    }
}

/// Element-wise array add assign
impl std::ops::AddAssign<Float3> for Float3 {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..3 { self.0[i] += rhs.0[i] }
    }
}

/// 3-element array add
impl std::ops::Add<Float3> for Float3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l + r))
    }
}

/// Element-wise array subtract assign
impl std::ops::SubAssign<Float3> for Float3 {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..3 { self.0[i] -= rhs.0[i] }
    }
}

/// 3-element array subtract
impl std::ops::Sub<Float3> for Float3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l - r))
    }
}

/// Scalar multiply
impl std::ops::Mul<f64> for Float3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::from_iter(self.0.iter().map(|x| x * rhs))
    }
}

/// Scalar multiply
impl std::ops::Mul<Float3> for f64 {
    type Output = Float3;
    fn mul(self, rhs: Float3) -> Float3 {
        Float3::from_iter(rhs.0.iter().map(|x| x * self))
    }
}

/// Scalar multiply assign
impl std::ops::MulAssign<f64> for Float3 {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..3 { self.0[i] *= rhs }
    }
}

/// Element-wise array multiplay
impl std::ops::Mul<Float3> for Float3 {
    type Output = Float3;
    fn mul(self, rhs: Float3) -> Float3 {
        Float3::from_iter(self.0.iter().zip(rhs.0.iter()).map(|(l, r)| l * r))
    }
}

/// Scalar divide assign
impl std::ops::DivAssign<f64> for Float3 {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..3 { self.0[i] /= rhs }
    }
}

/// Scalar devide
impl std::ops::Div<f64> for Float3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Float3::from_iter(self.0.iter().map(|x| x / rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_random() {
        for _ in 0..10 {
            println!("{:?}", Float3::random());
        }
    }

    #[test]
    fn test_vector() {
        assert_eq!(Float3([0.0, 0.0, 0.0]), Float3::zero());
        assert_eq!(Float3([1.0, 1.0, 1.0]), Float3::one());
        assert_eq!(Float3([5.0, 5.0, 5.0]), Float3::full(5.0));
        assert_eq!(Float3([0.0, 1.0, 0.42]), Float3::new(-1.2, 3.4, 0.42).saturate());
        for _ in 0..100 {
            let rnd = Float3::random_full();
            assert_eq!(rnd.x(), rnd.y());
            assert_eq!(rnd.x(), rnd.z());
            let Float3([x1, y1, z1]) = Float3::random();
            let Float3([x2, y2, z2]) = Float3::random();
            let v1 = Float3::new(x1, y1, z1);
            let v2 = Float3::new(x2, y2, z2);
            assert_eq!(Float3([-x1, -y1, -z1]), -v1);
            assert_eq!(x1 * x1 + y1 * y1 + z1 * z1, v1.length_squared());
            assert_eq!(x1 * x2 + y1 * y2 + z1 * z2, v1.dot(v2));
            assert_eq!(x1 * x2 + y1 * y2 + z1 * z2, v2.dot(v1));
            assert_eq!(Float3([x1.sqrt(), y1.sqrt(), z1.sqrt()]), v1.sqrt());
            assert_eq!(Float3([x1 + x2, y1 + y2, z1 + z2]), v1 + v2);
            assert_eq!(Float3([x1 - x2, y1 - y2, z1 - z2]), v1 - v2);
            assert_eq!(Float3([x1 * x2, y1 * y2, z1 * z2]), v1 * v2);
            assert_eq!(Float3([x1 * x2, y1 * x2, z1 * x2]), v1 * x2);
            assert_eq!(Float3([x1 * x2, y1 * x2, z1 * x2]), x2 * v1);
            assert_eq!(Float3([x1 / x2, y1 / x2, z1 / x2]), v1 / x2);
        }
    }

    #[test]
    fn test_color() {
        assert_eq!(Float3::new(1.0, 1.0, 0.0), Float3::from_hex(b"ffff00"));
        assert_eq!(Float3::new(0.0, 128.0 / 255.0, 1.0), Float3::from_hex(b"0080ff"));
        assert_eq!(Float3::new(0.0, 1.0, 1.0), Float3::from_rgb(0, 255, 255));
        assert_eq!(Float3::new(12.0 / 255.0, 96.0 / 255.0, 183.0 / 255.0), Float3::from_rgb(12, 96, 183));
    }
}
