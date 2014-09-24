use std::f32::consts::PI;
use std::ops::{Add,Sub,Mul};

#[deriving(PartialEq,Clone,Show)]
pub struct Vec3 {
  pub x:f32,
  pub y:f32,
  pub z:f32
}

impl Vec3 {
  pub fn new(x:f32, y:f32, z:f32) -> Vec3 {
    Vec3{x:x,y:y,z:z}
  }

  pub fn scale(&self, c:f32) -> Vec3 {
    Vec3 {
      x: self.x * c,
      y: self.y * c,
      z: self.z * c,
    }
  }

  pub fn dot(&self, other: &Vec3) -> f32 {
    (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
  }

  pub fn lerp(&self, other: &Vec3, t: f32) -> Vec3 {
    *self + (*other-*self).scale(t)
  }

  pub fn length_sq(&self) -> f32 {
    self.dot(self)
  }

  pub fn length(&self) -> f32 {
    self.length_sq().sqrt()
  }

  pub fn normalized(&self) -> Vec3 {
    self.scale( 1.0 / self.length() )
  }

  pub fn recip(&self) -> Vec3 {
    Vec3 {
      x: 1.0 / self.x,
      y: 1.0 / self.y,
      z: 1.0 / self.z,
    }
  }

  pub fn min(&self, other: &Vec3) -> Vec3 {
    Vec3 {
      x: self.x.min( other.x ),
      y: self.y.min( other.y ),
      z: self.z.min( other.z ),
    }
  }

  pub fn max(&self, other: &Vec3) -> Vec3 {
    Vec3 {
      x: self.x.max( other.x ),
      y: self.y.max( other.y ),
      z: self.z.max( other.z ),
    }
  }

  pub fn cross(&self, other: &Vec3) -> Vec3 {
    Vec3 {
      x: (self.y * other.z) - (other.y * self.z),
      y: (self.z * other.x) - (other.z * self.x),
      z: (self.x * other.y) - (other.x * self.y),
    }
  }
}

impl Add<Vec3, Vec3> for Vec3 {
  fn add(&self, rhs: &Vec3) -> Vec3 {
    Vec3 {
      x: self.x + rhs.x,
      y: self.y + rhs.y,
      z: self.z + rhs.z,
    }
  }
}

impl Sub<Vec3, Vec3> for Vec3 {
  fn sub(&self, rhs: &Vec3) -> Vec3 {
    Vec3 {
      x: self.x - rhs.x,
      y: self.y - rhs.y,
      z: self.z - rhs.z,
    }
  }
}

impl Mul<Vec3, Vec3> for Vec3 {
  fn mul(&self, rhs: &Vec3) -> Vec3 {
    Vec3 {
      x: self.x * rhs.x,
      y: self.y * rhs.y,
      z: self.z * rhs.z,
    }
  }
}

pub struct Mtx33 {
  r0:Vec3,
  r1:Vec3,
  r2:Vec3,
}

impl Mtx33 {
  pub fn transform(&self, v: Vec3) -> Vec3 {
    Vec3::new(
      self.r0.dot( &v ),
      self.r1.dot( &v ),
      self.r2.dot( &v )
    )
  }

  pub fn transposed(&self) -> Mtx33 {
    Mtx33 {
      r0: Vec3::new( self.r0.x, self.r1.x, self.r2.x ),
      r1: Vec3::new( self.r0.y, self.r1.y, self.r2.y ),
      r2: Vec3::new( self.r0.z, self.r1.z, self.r2.z ),
    }
  }
}

impl Mul<Mtx33,Mtx33> for Mtx33 {
  fn mul(&self, rhs: &Mtx33) -> Mtx33 {
    let rhs = rhs.transposed();
    Mtx33 {
      r0: Vec3::new( self.r0.dot(&rhs.r0), self.r0.dot(&rhs.r1), self.r0.dot(&rhs.r2) ),
      r1: Vec3::new( self.r1.dot(&rhs.r0), self.r1.dot(&rhs.r1), self.r1.dot(&rhs.r2) ),
      r2: Vec3::new( self.r2.dot(&rhs.r0), self.r2.dot(&rhs.r1), self.r2.dot(&rhs.r2) ),
    }
  }
}

pub struct Ray {
  pub origin: Vec3,
  pub dir: Vec3,
}
pub struct Triangle {
  pub p1: Vec3,
  pub p2: Vec3,
  pub p3: Vec3,
}
pub struct HitResult {
  pub barycentric: Vec3,
  pub t: f32
}

impl Ray {
  #[inline(always)]
  pub fn intersect(&self, t: &Triangle) -> Option<HitResult> {
    let e1 = t.p2 - t.p1;
    let e2 = t.p3 - t.p1;
    let s1 = self.dir.cross(&e2);
    let divisor = s1.dot(&e1);
    if divisor == 0.0 {
      return None;
    }

    // compute first barycentric coordinate
    let inv_divisor = 1.0 / divisor;
    let d = self.origin - t.p1;

    let b1 = d.dot(&s1) * inv_divisor;
    if b1 < 0.0 || b1 > 1.0 {
      return None;
    }

    // and second barycentric coordinate
    let s2 = d.cross(&e1);
    let b2 = self.dir.dot(&s2) * inv_divisor;

    if b2 < 0.0 || b1+b2 > 1.0 {
      return None; // outside triangle
    }

    let t = e2.dot(&s2) * inv_divisor;
    if t < 0.0 {
      None // behind viewer
    } else {
      Some( HitResult{ barycentric: Vec3::new(b1, b2, 1.0-b1-b2), t: t} )
    }
  }
  #[inline(always)]
  pub fn bb_check(&self, max_dist: f32, bbox: BoundingBox ) -> bool {
    let inv_dir = self.dir.recip();
    let (tx1,tx2,ty1,ty2,tz1,tz2) = (
      (bbox.min.x - self.origin.x)*inv_dir.x,
      (bbox.max.x - self.origin.x)*inv_dir.x,
      (bbox.min.y - self.origin.y)*inv_dir.y,
      (bbox.max.y - self.origin.y)*inv_dir.y,
      (bbox.min.z - self.origin.z)*inv_dir.z,
      (bbox.max.z - self.origin.z)*inv_dir.z
      );

    let (minx, maxx) = (tx1.min(tx2), tx1.max(tx2));
    let (miny, maxy) = (ty1.min(ty2), ty1.max(ty2));
    let (minz, maxz) = (tz1.min(tz2), tz1.max(tz2));

    let tmin = minx.max( miny.max( minz ) );
    let tmax = maxx.min( maxy.min( maxz ) );
    tmax >= 0.0 && tmin <= tmax && tmin <= max_dist
  }
}


pub struct BoundingBox {
  pub min: Vec3,
  pub max: Vec3,
}

// Gives a cosine hemisphere sample from two uniform f32s
// in [0,1) range.
#[inline(always)]
pub fn cosine_hemisphere_sample( u: f32, v: f32 ) -> Vec3 {
  let r_sqrt = u.sqrt();
  let theta = 2f32 * PI * v;
  Vec3::new( r_sqrt*theta.cos(), (1f32-u).sqrt(), r_sqrt*theta.sin() )
}

#[inline(always)]
pub fn rotate_to_up( up_vec: Vec3 ) -> Mtx33 {
  let perp = if up_vec == Vec3::new(0f32,1f32,0f32) { Vec3::new(1f32,0f32,0f32) } else { Vec3::new(0f32,1f32,0f32) };
  let right = up_vec.cross( &perp );
  let fwd = right.cross( &up_vec );
  Mtx33{ r0: right, r1: up_vec, r2: fwd }.transposed()
}

#[inline(always)]
pub fn rotate_y(theta: f32) -> Mtx33{
  let ct = theta.cos();
  let st = theta.sin();
  Mtx33{ r0: Vec3::new(ct,0f32,st), r1: Vec3::new(0f32,1f32,0f32), r2: Vec3::new(-st, 0f32, ct) }
}

#[test]
pub fn dot_test()
{
  let v1 = Vec3::new(1.0,2.0,3.0);
  let v2 = Vec3::new(4.0,5.0,6.0);
  let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
  assert!(v1.dot(&v2) == expected);
}

#[test]
pub fn cross_test()
{
  let (x1,x2,y1,y2,z1,z2) = (1.0,4.0,2.0,5.0,3.0,6.0);
  let v1 = Vec3::new(x1,y1,z1);
  let v2 = Vec3::new(x2,y2,z2);
  let expected = Vec3::new(
    (y1 * z2) - (y2 * z1),
    (z1 * x2) - (z2 * x1),
    (x1 * y2) - (x2 * y1)
  );
  let result = v1.cross(&v2);
  assert!( expected.x == result.x );
  assert!( expected.y == result.y );
  assert!( expected.z == result.z );
}

#[test]
pub fn intersection_test()
{
  let ray = Ray{
    origin: Vec3::new(0f32, 0f32, 0f32),
    dir: Vec3::new(0.0f32,0.0f32,-1.0f32),
  };
  let tri = Triangle{
    p1: Vec3::new(-1f32, -1f32, -1f32),
    p2: Vec3::new(1f32, -1f32, -1f32),
    p3: Vec3::new(0f32, 2f32, -1f32),
  };

  assert!(ray.intersect(&tri).is_some());
}
