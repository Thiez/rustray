use std::*;

#[deriving(Eq,Clone)]
pub struct vec3 {
    x:f32,
    y:f32,
    z:f32
}

pub struct mtx33 {
    r0:vec3,
    r1:vec3,
    r2:vec3
}

#[inline(always)]
pub fn vec3(x:f32, y:f32, z:f32) -> vec3{
    vec3 {x:x, y:y, z:z}
}

#[inline(always)]
pub fn dot(a:vec3, b:vec3) -> f32 {
    a.x*b.x + a.y*b.y + a.z*b.z
}

#[inline(always)]
pub fn lerp(a:vec3, b:vec3, t:f32) -> vec3{
    add( a, scale(sub(b,a),t) )
}

#[inline(always)]
pub fn scale(v:vec3, s:f32) -> vec3 {
    vec3 { x:v.x*s, y:v.y*s, z:v.z*s }
}

#[inline(always)]
pub fn length_sq(v:vec3) -> f32 {
    dot(v,v)
}

#[inline(always)]
pub fn length(v:vec3) -> f32 {
    f32::sqrt(length_sq(v))
}

#[inline(always)]
pub fn normalized(v:vec3) -> vec3 {
    scale(v, 1.0f32 / length(v))
}

#[inline(always)]
pub fn recip(a:vec3) -> vec3{
    vec3(1f32/a.x, 1f32/a.y, 1f32/a.z)
}


#[inline(always)]
pub fn mul(a:vec3, b:vec3) -> vec3{
    vec3(a.x*b.x, a.y*b.y, a.z*b.z)
}

#[inline(always)]
pub fn add(a:vec3, b:vec3) -> vec3 {
    vec3 {x:a.x+b.x, y:a.y+b.y, z:a.z+b.z}
}

#[inline(always)]
pub fn sub(a:vec3, b:vec3) -> vec3 {
    add(a, scale(b, -1.0f32))
}

#[inline(always)]
pub fn cross(a:vec3, b:vec3) -> vec3 {
    vec3( a.y*b.z - b.y*a.z,
          a.z*b.x - b.z*a.x,
          a.x*b.y - b.x*a.y)
}

#[inline(always)]
pub fn min(a: vec3, b: vec3) -> vec3 {
    vec3( f32::fmin(a.x,b.x), f32::fmin(a.y, b.y), f32::fmin(a.z, b.z) )
}

#[inline(always)]
pub fn max(a: vec3, b: vec3) -> vec3 {
    vec3( f32::fmax(a.x,b.x), f32::fmax(a.y, b.y), f32::fmax(a.z, b.z) )
}

pub struct Ray { origin:vec3, dir:vec3 }
pub struct Triangle { p1: vec3, p2: vec3, p3: vec3 }
pub struct HitResult { barycentric: vec3, t: f32 }

pub impl Ray {
    #[inline(always)]
    fn intersect(&self, t: &Triangle) -> Option<HitResult> {
        let e1 = sub(t.p2,t.p1);
        let e2 = sub(t.p3,t.p1);
        let s1 = cross(self.dir,e2);
        let divisor = dot(s1,e1);

        if divisor == 0f32 {
            return option::None;
        }

        // compute first barycentric coordinate
        let inv_divisor = 1.0f32 / divisor;
        let d = sub(self.origin,t.p1);

        let b1 = dot(d, s1) * inv_divisor;
        if b1 < 0.0f32 || b1 > 1.0f32 {
            return option::None;
        }

        // and second barycentric coordinate
        let s2 = cross(d,e1);
        let b2 = dot(self.dir,s2) * inv_divisor;
        if b2 < 0.0f32 || b1+b2 > 1.0f32 {
            return option::None; // outside triangle
        }

        let t = dot(e2,s2) * inv_divisor;
        if t < 0.0f32 {
            return option::None; // behind viewer
        }
        option::Some( HitResult{barycentric: vec3(b1, b2, 1.0f32-b1-b2), t: t} )
    }
    #[inline(always)]
    fn aabb_check(&self, max_dist: f32, box: aabb ) -> bool {
        let inv_dir = recip(self.dir);
        let (tx1,tx2,ty1,ty2,tz1,tz2) = (
            (box.min.x - self.origin.x)*inv_dir.x,
            (box.max.x - self.origin.x)*inv_dir.x,
            (box.min.y - self.origin.y)*inv_dir.y,
            (box.max.y - self.origin.y)*inv_dir.y,
            (box.min.z - self.origin.z)*inv_dir.z,
            (box.max.z - self.origin.z)*inv_dir.z
        );

        let (minx, maxx) = (f32::fmin(tx1,tx2), f32::fmax(tx1,tx2));
        let (miny, maxy) = (f32::fmin(ty1,ty2), f32::fmax(ty1,ty2));
        let (minz, maxz) = (f32::fmin(tz1,tz2), f32::fmax(tz1,tz2));

        let tmin = f32::fmax(minx, f32::fmax(miny, minz));
        let tmax = f32::fmin(maxx, f32::fmin(maxy, maxz));

        tmax >= 0f32 && tmin <= tmax && tmin <= max_dist
    }
}


pub struct aabb {
    min: vec3,
    max: vec3
}

// Gives a cosine hemisphere sample from two uniform f32s
// in [0,1) range.
#[inline(always)]
pub fn cosine_hemisphere_sample( u: f32, v: f32 ) -> vec3 {
    let r_sqrt = f32::sqrt(u);
    let theta = 2f32 * f32::consts::pi * v;
    vec3( r_sqrt*f32::cos(theta), f32::sqrt(1f32-u), r_sqrt*f32::sin(theta))
}

#[inline(always)]
pub fn rotate_to_up( up_vec: vec3 ) -> mtx33 {
    let perp = if up_vec == vec3(0f32,1f32,0f32) { vec3(1f32,0f32,0f32) } else { vec3(0f32,1f32,0f32) };
    let right = cross( up_vec, perp );
    let fwd = cross(right, up_vec );
    transposed( mtx33{ r0: right, r1: up_vec, r2: fwd } )
}

#[inline(always)]
pub fn rotate_y(theta: f32) -> mtx33{
    let ct = f32::cos(theta);
    let st = f32::sin(theta);
    mtx33{ r0: vec3(ct,0f32,st), r1: vec3(0f32,1f32,0f32), r2: vec3(-st, 0f32, ct) }
}

#[inline(always)]
pub fn transform( m: mtx33, v: vec3 ) -> vec3 {
    vec3( dot( m.r0, v ), dot( m.r1, v ), dot( m.r2, v ) )
}

#[inline(always)]
pub fn mul_mtx33( a: mtx33, b: mtx33 ) -> mtx33 {
    let b_t = transposed(b);
    mtx33 {   r0: vec3( dot(a.r0,b_t.r0), dot(a.r0,b_t.r1), dot(a.r0,b_t.r2) ),
        r1: vec3( dot(a.r1,b_t.r0), dot(a.r1,b_t.r1), dot(a.r1,b_t.r2) ),
        r2: vec3( dot(a.r2,b_t.r0), dot(a.r2,b_t.r1), dot(a.r2,b_t.r2) ) }
}

#[inline(always)]
pub fn transposed( m: mtx33 ) -> mtx33 {
    mtx33 {   r0: vec3( m.r0.x, m.r1.x, m.r2.x ),
        r1: vec3( m.r0.y, m.r1.y, m.r2.y ),
        r2: vec3( m.r0.z, m.r1.z, m.r2.z ) }
}

#[test]
pub fn intersection_test()
{
    let ray = Ray{ origin: vec3(0f32, 0f32, 0f32), dir: vec3(0.0f32,0.0f32,-1.0f32) };
    let tri = Triangle{  p1: vec3(-1f32, -1f32, -1f32),
                    p2: vec3(1f32, -1f32, -1f32),
                    p3: vec3(0f32, 2f32, -1f32) };

    assert!(option::is_some( &ray.intersect(&tri) ));
}
