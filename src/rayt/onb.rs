use crate::rayt::*;

pub struct ONB {
    axis: [Vec3; 3],
}

impl ONB {
    pub fn new(n: Vec3) -> Self {
        let w = n.normalize();
        let v = if w.x().abs() > 0.9 {
            w.cross(Vec3::yaxis()).normalize()
        } else {
            w.cross(Vec3::xaxis()).normalize()
        };
        let u = w.cross(v);
        Self { axis: [u, v, w] }
    }

    pub fn u(&self) -> Vec3 { self.axis[0] }
    pub fn v(&self) -> Vec3 { self.axis[1] }
    pub fn w(&self) -> Vec3 { self.axis[2] }

    pub fn local(&self, v: Vec3) -> Vec3 {
        self.axis[0] * v.x() + self.axis[1] * v.y() + self.axis[2] * v.z()
    }
}
