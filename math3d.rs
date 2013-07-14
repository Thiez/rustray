use std::f32::consts::pi;
use std::unstable::simd::{f32x4};

pub struct mtx33 {
    r0:f32x4,
    r1:f32x4,
    r2:f32x4,
}

#[inline(always)]
pub fn select(a: f32x4, n: uint) -> f32 {
    match n {
        0   => {
                let f32x4(t,_,_,_) = a;
                t
            }
        1   => {
                let f32x4(_,t,_,_) = a;
                t
            }
        2   => {
                let f32x4(_,_,t,_) = a;
                t
            }
        _ => 0.0
    }
}

#[inline(always)]
pub fn scale(a: f32x4, c:f32) -> f32x4 {
    a * f32x4(c,c,c,c)
}

pub fn dot(a:f32x4, b:f32x4) -> f32 {
    let f32x4(x,y,z,_) = a * b;
    x+y+z
}

#[inline(always)]
pub fn lerp(a: f32x4, b:f32x4, t:f32) -> f32x4 {
    a + scale(b-a, t)
}

#[inline(always)]
pub fn length_sq(a:f32x4) -> f32 {
    dot(a,a)
}

#[inline(always)]
pub fn length(a:f32x4) -> f32 {
    length_sq(a).sqrt()
}

#[inline(always)]
pub fn normalized(a:f32x4) -> f32x4 {
    scale(a, 1.0f32/length(a))
}

#[inline(always)]
pub fn recip(a: f32x4) -> f32x4 {
    let f32x4(x,y,z,_) = f32x4(1.0,1.0,1.0,0.0) / a;
    f32x4(x,y,z,0.0)
}

#[inline(always)]
pub fn cross(a: f32x4, b: f32x4) -> f32x4 {
    let f32x4(a1,a2,a3,_) = a;
    let f32x4(b1,b2,b3,_) = b;
    f32x4(a2*b3 - b2*a3, a3*b1-b3*a1, a1*b2 - b1*a2,0.0)
}

#[inline(always)]
pub fn min(a: f32x4, b: f32x4) -> f32x4 {
    let f32x4(a1,a2,a3,_) = a;
    let f32x4(b1,b2,b3,_) = b;
    f32x4(a1.min(&b1), a2.min(&b2), a3.min(&b3),0.0)
}

#[inline(always)]
pub fn max(a: f32x4, b: f32x4) -> f32x4 {
    let f32x4(a1,a2,a3,_) = a;
    let f32x4(b1,b2,b3,_) = b;
    f32x4(a1.max(&b1), a2.max(&b2), a3.max(&b3),0.0)
}

pub struct Ray {
    origin: f32x4,
    dir: f32x4,
}

pub struct Triangle {
    p1: f32x4,
    p2: f32x4,
    p3: f32x4,
}

pub struct HitResult {
    barycentric: f32x4,
    t: f32,
}

pub struct aabb {
    min: f32x4,
    max: f32x4,
}

impl Ray {
    #[inline]
    pub fn intersect(&self, t: &Triangle) -> Option<HitResult> {
        let e1 = t.p2 - t.p1;
        let e2 = t.p3 - t.p1;
        let s1 = self.dir * e2;
        let divisor = dot(s1,e2);
        
        if divisor == 0.0 {
            return None;
        }
        
        // compute first barycentric coordinate
        let inv_divisor = 1.0 / divisor;
        let d = self.origin - t.p1;
        
        let b1 = dot(d,s1) * inv_divisor;
        if b1 < 0.0 || b1 > 1.0 {
            return None;
        }
        
        // and second barycentric coordinate
        let s2 = cross(d,e1);
        let b2 = dot(self.dir,s2) * inv_divisor;
        
        if b2 < 0.0 || b1+b2 > 1.0 {
            return None;
        }
        
        let t = dot(e2,s2) * inv_divisor;
        if t < 0.0 {
            None
        } else {
            Some( HitResult{ barycentric: f32x4(b1,b2,1.0-b1-b2,0.0), t: t } )
        }
    }
    #[inline]
    pub fn aabb_check(&self, max_dist: f32, box: aabb) -> bool {
        let inv_dir = recip(self.dir);
        let t1s = (box.min - self.origin) * inv_dir;
        let t2s = (box.max - self.origin) * inv_dir;
        
        let f32x4(minx, miny, minz, _) = min(t1s,t2s);
        let f32x4(maxx, maxy, maxz, _) = max(t1s,t2s);
        
        let tmin = minx.max( &miny.max( &minz ) );
        let tmax = maxx.min( &maxy.min( &maxz ) );
        
        tmax >= 0.0 && tmin <= tmax && tmin <= max_dist
    }
}

// Gives a cosine hemisphere sample from two uniform f32s
// in [0,1) range.
#[inline(always)]
pub fn cosine_hemisphere_sample( u: f32, v: f32) -> f32x4 {
    let r_sqrt = u.sqrt();
    let theta = 2.0 * pi * v;
    
    scale( f32x4(theta.cos(), 0.0, theta.sin(), 0.0), r_sqrt )
        + f32x4(0.0, (1.0-u).sqrt(), 0.0, 0.0)
}

#[inline(always)]
pub fn rotate_to_up( up_vec: f32x4 ) -> mtx33 {
    let perp = match up_vec {
        f32x4(0.0,1.0,0.0,_)    => f32x4(1.0, 0.0, 0.0, 0.0),
        _                       => f32x4(0.0, 1.0, 0.0, 0.0),
    };
    let right = cross(up_vec, perp);
    let fwd = cross(right, up_vec);
    transposed( mtx33{ r0: right, r1: up_vec, r2: fwd } )
}

#[inline(always)]
pub fn rotate_y(theta: f32) -> mtx33 {
    let ct = theta.cos();
    let st = theta.sin();
    mtx33 {
        r0: f32x4(ct,  0.0, st,  0.0),
        r1: f32x4(0.0, 1.0, 0.0, 0.0),
        r2: f32x4(-st, 0.0, ct,  0.0),
    }
}

#[inline(always)]
pub fn transform( m: mtx33, v: f32x4 ) -> f32x4 {
    f32x4( dot( m.r0, v ), dot( m.r1, v ), dot( m.r2, v ), 0.0 )
}

#[inline(always)]
pub fn mul_mtx33( a: mtx33, b: mtx33 ) -> mtx33 {
    let b = transposed(b);
    mtx33 {
        r0: f32x4( dot(a.r0,b.r0), dot(a.r0, b.r1 ), dot(a.r0, b.r2), 0.0 ),
        r1: f32x4( dot(a.r1,b.r0), dot(a.r1, b.r1 ), dot(a.r1, b.r2), 0.0 ),
        r2: f32x4( dot(a.r2,b.r0), dot(a.r2, b.r1 ), dot(a.r2, b.r2), 0.0 ),
    }
}

#[inline(always)]
pub fn transposed( m: mtx33 ) -> mtx33 {
    let f32x4(a1,a2,a3,_) = m.r0;
    let f32x4(b1,b2,b3,_) = m.r1;
    let f32x4(c1,c2,c3,_) = m.r2;
    mtx33 {
        r0: f32x4(a1, b1, c1, 0.0),
        r1: f32x4(a2, b2, c2, 0.0),
        r2: f32x4(a3, b3, c3, 0.0),
    }
}

