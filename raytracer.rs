use super::math3d::{Vec3,Ray,Triangle,HitResult,cosine_hemisphere_sample,rotate_to_up,rotate_y};
use super::consts;
use super::concurrent;
use super::model;
use std::{io,f32,uint};
use std::vec::{Vec};
use std::cmp::{min};
use std::num::Float;
use rand::{Rng,task_rng,TaskRng};

pub struct Color {
  pub r:u8,
  pub g:u8,
  pub b:u8,
}

trait ToColor {
  fn toColor(self)->Color;
}

impl ToColor for (f32,f32,f32) {
  fn toColor(self)->Color {
    let (r,g,b) = self;
    Color {
      r: r as u8,
      g: g as u8,
      b: b as u8,
    }
  }
}

struct PixelCoords {
  x: uint,
  y: uint,
}

struct PixelIterator {
  width: uint,
  height: uint,
  row: uint,
  col: uint,
}

impl PixelIterator {
  fn new(width: uint, height: uint) -> PixelIterator {
    PixelIterator {
      width: width,
      height: height,
      row: 0,
      col: uint::MAX, // we overflow to zero on first call to next()
    }
  }
}

impl Iterator<PixelCoords> for PixelIterator {
  fn next(&mut self) -> Option<PixelCoords> {
    self.col += 1;
    if self.col >= self.width {
      self.col = 0;
      self.row += 1;
      if self.row >= self.height {
        return None;
      }
    }
    Some(PixelCoords {
      x: self.col,
      y: self.row
    })
  }
}

#[inline]
fn get_ray( horizontalFOV: f32, width: uint, height: uint, x: uint, y: uint, sample_jitter : (f32,f32)) -> Ray {
  let (jitterx,jittery) = sample_jitter;
  let dirx = (x as f32) - ((width/2) as f32) + jitterx;
  let diry = -((y as f32) - ((height/2) as f32)) + jittery;
  let dirz = -((width/2u) as f32) / (horizontalFOV*0.5).tan();
  Ray{
    origin: Vec3::new(0.0, 0.0, 1.0),
    dir: Vec3::new( dirx, diry, dirz).normalized()
  }
}

#[deriving(Clone)]
struct RandEnv {
  floats: ~[f32],
  disk_samples: ~[(f32,f32)],
  hemicos_samples: ~[Vec3]
}

fn get_rand_env() -> RandEnv {
  let mut gen = task_rng();

  let disk_samples = Vec::from_fn(513u, |_x| {
    // compute random position on light disk
    let r_sqrt = gen.gen::<f32>().sqrt();
    let theta = gen.gen::<f32>() * 2.0 * f32::consts::PI;
    (r_sqrt * theta.cos(), r_sqrt*theta.sin())
  });

  let mut hemicos_samples = Vec::with_capacity(consts::NUM_GI_SAMPLES_SQRT * consts::NUM_GI_SAMPLES_SQRT);

  for x in range(0, consts::NUM_GI_SAMPLES_SQRT) {
    for y in range(0, consts::NUM_GI_SAMPLES_SQRT) {
      let (u,v) = (
        ( x as f32 + gen.gen() ) / (consts::NUM_GI_SAMPLES_SQRT as f32),
        ( y as f32 + gen.gen() ) / (consts::NUM_GI_SAMPLES_SQRT as f32)
      );
      hemicos_samples.push(cosine_hemisphere_sample(u,v));
    }
  };

  RandEnv {
    floats: Vec::from_fn(513u, |_x| gen.gen() ).move_iter().collect(),
    disk_samples: disk_samples.move_iter().collect(),
    hemicos_samples: hemicos_samples.move_iter().collect()
  }
}

#[inline]
fn sample_disk( rnd: &RandEnv, num: uint, body: |f32,f32|->() ){
  let mut rng = task_rng();
  if num == 1 {
    body(0.0,0.0);
  } else {
    let mut ix = rng.gen::<uint>() % rnd.disk_samples.len(); // start at random location
    for _ in range(0,num) {
      let (u,v) = rnd.disk_samples[ix];
      body(u,v);
      ix = (ix + 1) % rnd.disk_samples.len();
    };
  }
}

struct Stratified2dIterator<'rand> {
  rnd: &'rand RandEnv,
  rng: TaskRng,
  mSamples: uint,
  nSamples: uint,
  mIndex: uint,
  nIndex: uint,
  offset: uint,
}

impl<'rand> Stratified2dIterator<'rand> {
  fn new(rnd: &'rand RandEnv, mSamples: uint, nSamples: uint) -> Stratified2dIterator<'rand> {
    let mut rng = task_rng();
    let offset = rng.gen::<uint>();
    Stratified2dIterator {
      rnd: rnd,
      rng: rng,
      mSamples: mSamples,
      nSamples: nSamples,
      mIndex: 0,
      nIndex: uint::MAX, // overflows on first call to next()
      offset: offset,
    }
  }
}

impl<'rand> Iterator<(f32,f32)> for Stratified2dIterator<'rand> {
  fn next(&mut self) -> Option<(f32,f32)> {
    self.nIndex += 1;
    if self.nIndex >= self.nSamples {
      self.nIndex = 0;
      self.mIndex += 1;
      if self.mIndex >= self.mSamples {
        return None;
      }
    }
    let offset = self.offset + self.nSamples * self.nIndex;
    if self.nSamples == 1 {
      Some((1.0,1.0))
    } else {
      let len = self.rnd.floats.len();
      let r1 = (self.mIndex as f32 + self.rnd.floats[offset % len]) / (self.mSamples as f32);
      let r2 = (self.nIndex as f32 + self.rnd.floats[(offset + 1) % len]) / (self.nSamples as f32);
      self.offset += 2;
      Some((r1,r2))
    }
  }
}

#[inline(always)]
fn sample_cosine_hemisphere( rnd: &RandEnv, n: Vec3, body: |Vec3|->() ) {
  let mut rng = task_rng();
  let rot_to_up = rotate_to_up(n);
  let random_rot = rotate_y( rnd.floats[ rng.gen::<uint>() % rnd.floats.len() ] ); // random angle about y
  let m = rot_to_up * random_rot;
  for s in rnd.hemicos_samples.iter() {
    body(m.transform(*s));
  }
}

#[inline(always)]
fn get_triangle( m : &model::Polysoup, ix : uint ) -> Triangle{
  Triangle{
    p1: m.vertices[ m.indices[ix*3] ],
    p2: m.vertices[ m.indices[ix*3+1] ],
    p3: m.vertices[ m.indices[ix*3+2] ]
  }
}

#[inline(always)]
fn clamp( x: f32, lo: f32, hi: f32 ) -> f32 {
  if x < lo { lo } else if x > hi { hi } else { x }
}

#[inline]
fn trace_kd_tree(
  polys: &model::Polysoup,
  kd_tree_nodes: &[model::KdTreeNode],
  kd_tree_root: uint,
  r: &Ray,
  inv_dir: Vec3,
  inmint: f32,
  inmaxt: f32 )
-> Option<(HitResult, uint)> {

  let mut res : Option<(HitResult, uint)> = None;
  let mut closest_hit = inmaxt;

  let mut stack : ~[(uint, f32, f32)] = ~[];
  let mut mint = inmint;
  let mut maxt = inmaxt;
  let mut cur_node = kd_tree_root;

  loop {
    // skip any nodes that have been superceded
    // by a closer hit.
    while mint >= closest_hit {
      if stack.len() > 0 {
        let (n,mn,mx) = stack.pop().unwrap();
        cur_node = n;
        mint = mn;
        maxt = mx;
      } else {
        return res;
      }
    }

    match kd_tree_nodes[cur_node] {
      model::KdLeaf(tri_begin, tri_count) => {
        let mut tri_index : u32 = tri_begin;
        while tri_index < tri_begin+tri_count {

          let t = &get_triangle( polys, tri_index as uint );
          let new_hit_result = r.intersect(t);

          match (res, new_hit_result){
            (None, Some(hr)) => {
              res = Some((hr,tri_index as uint));
              closest_hit = hr.t;
            }
            (Some((hr1,_)), Some(hr2)) if hr1.t > hr2.t => {
              res = Some((hr2,tri_index as uint));
              closest_hit = hr2.t;
            }
            _ => {}
          }
          tri_index += 1;
        }

        if stack.len() > 0 {
          let (n,mn,mx) = stack.pop().unwrap();
          cur_node = n;
          mint = mn;
          maxt = mx;
        } else {
          return res;
        }
      }
      model::KdNode(axis, splitter, right_tree) => {
        // find the scalar direction/origin for the current axis
        let (inv_dir_scalar, origin) = match axis {
          model::AxisX => { (inv_dir.x, r.origin.x) }
          model::AxisY => { (inv_dir.y, r.origin.y) }
          model::AxisZ => { (inv_dir.z, r.origin.z) }
        };
        // figure out which side of the spliting plane the ray origin is
        // i.e. which child we need to test first.
        let (near,far) = if origin < splitter || (origin == splitter && inv_dir_scalar >= 0.0) {
          ((cur_node+1) as uint,right_tree as uint)
        } else {
          (right_tree as uint, (cur_node+1) as uint)
        };
        // find intersection with plane
        // origin + dir*plane_dist = splitter
        let plane_dist = (splitter - origin) * inv_dir_scalar;

        if plane_dist > maxt || plane_dist <= 0.0 {
          cur_node = near;
        } else if plane_dist < mint {
          cur_node = far;
        } else{
          stack.push((far, plane_dist, maxt) );
          cur_node = near;
          maxt = plane_dist;
        }
      }
    }
  }
}

#[inline]
fn trace_kd_tree_shadow(
  polys: &model::Polysoup,
  kd_tree_nodes: &[model::KdTreeNode],
  kd_tree_root: uint,
  r: &Ray,
  inv_dir: Vec3,
  inmint: f32,
  inmaxt: f32 )
-> bool {

  let mut stack : ~[(u32, f32, f32)] = ~[];
  let mut mint = inmint;
  let mut maxt = inmaxt;
  let mut cur_node = kd_tree_root;
  loop {

    match kd_tree_nodes[cur_node] {
      model::KdLeaf(tri_begin, tri_count) => {

        let mut tri_index = tri_begin;
        while tri_index < tri_begin + tri_count {
          let t = &get_triangle( polys, tri_index as uint);
          if r.intersect(t).is_some() {
            return true;
          }
          tri_index += 1;
        }
        if stack.len() > 0 {
          let (n,mn,mx) = stack.pop().unwrap();
          cur_node = n as uint;
          mint = mn;
          maxt = mx;
        } else {
          return false;
        }
      }
      model::KdNode(axis, splitter, right_tree) => {

        // find the scalar direction/origin for the current axis
        let (inv_dir_scalar, origin) = match axis {
          model::AxisX => { (inv_dir.x, r.origin.x) }
          model::AxisY => { (inv_dir.y, r.origin.y) }
          model::AxisZ => { (inv_dir.z, r.origin.z) }
        };

        // figure out which side of the spliting plane the ray origin is
        // i.e. which child we need to test first.
        let (near,far) = if origin < splitter || (origin == splitter && inv_dir_scalar >= 0.0) {
          ((cur_node+1) as u32,right_tree)
        } else {
          (right_tree, (cur_node+1) as u32)
        };

        // find intersection with plane
        // origin + dir*t = splitter
        let plane_dist = (splitter - origin) * inv_dir_scalar;

        if plane_dist > maxt || plane_dist < 0.0 {
          cur_node = near as uint;
        } else if plane_dist < mint {
          cur_node = far as uint;
        } else{
          stack.push((far, plane_dist, maxt));
          cur_node = near as uint;
          maxt = plane_dist;
        }
      }
    }
  }
}

#[inline]
fn trace_soup( polys: &model::Polysoup, r: &Ray) -> Option<(HitResult, uint)>{

  let mut res : Option<(HitResult, uint)> = None;

  for tri_ix in range(0, polys.indices.len() / 3) {
    let tri = &get_triangle( polys, tri_ix);

    let new_hit = r.intersect(tri);

    match (res, new_hit) {
      (None,Some(hit)) => {
        res = Some((hit, tri_ix));
      }
      (Some((old_hit,_)), Some(hit))
        if hit.t < old_hit.t && hit.t > 0.0 => {
          res = Some((hit, tri_ix));
        }
      _ => {}
    }
  }

  return res;
}

#[deriving(Clone)]
struct Light {
  pos: Vec3,
  strength: f32,
  radius: f32,
  color: Vec3
}

impl Light {
  fn new(pos: Vec3, strength: f32, radius: f32, color: Vec3) -> Light {
    Light {
      pos: pos,
      strength: strength,
      radius: radius,
      color: color,
    }
  }
}

#[inline(always)]
fn direct_lighting( lights: &[Light], pos: Vec3, n: Vec3, view_vec: Vec3, rnd: &RandEnv, depth: uint, occlusion_probe: |Vec3| -> bool ) -> Vec3 {

  let mut direct_light = Vec3::new(0.0,0.0,0.0);
  for l in lights.iter() {

    // compute shadow contribution
    let mut shadow_contrib = 0.0;
    let num_samples = match depth { 0 => consts::NUM_LIGHT_SAMPLES, _ => 1 };    // do one tap in reflections and GI

    let rot_to_up = rotate_to_up((pos - l.pos).normalized());
    let shadow_sample_weight = 1.0 / (num_samples as f32);
    sample_disk(rnd ,num_samples, |u,v| {        // todo: stratify this

      // scale and rotate disk sample, and position it at the light's location
      let sample_pos = l.pos + rot_to_up.transform( Vec3::new(u*l.radius,0f32,v*l.radius) );

      if !occlusion_probe( sample_pos - pos ) {
        shadow_contrib += shadow_sample_weight;
      }
    });

    let light_vec = l.pos - pos;
    let light_contrib =
      if shadow_contrib == 0.0 {
        Vec3::new(0.0, 0.0, 0.0)
      } else {

        let light_vec_n = light_vec.normalized();
        let half_vector = (light_vec_n + view_vec).normalized();

        let s = n.dot(&half_vector);
        let specular = s.powf(&175.0);

        let atten = shadow_contrib*l.strength*(1.0/light_vec.length_sq() + specular*0.05);

        let intensity = atten * n.dot( &light_vec_n );

        l.color.scale(intensity)
      };

    direct_light = direct_light + light_contrib;
  }

  direct_light
}

#[inline]
fn shade(
  pos: Vec3, n: Vec3, n_face: Vec3, r: &Ray, color: Vec3, reflectivity: f32, lights: &[Light], rnd: &RandEnv, depth: uint,
  occlusion_probe: |Vec3| -> bool,
  color_probe: |Vec3| -> Vec3 ) -> Vec3 {

  let view_vec = (r.origin - pos).normalized();

  // pass in n or n_face for smooth/flat shading
  let shading_normal = if consts::USE_SMOOTH_NORMALS_FOR_DIRECT_LIGHTING { n } else { n_face };

  let direct_light = direct_lighting(lights, pos, shading_normal, view_vec, rnd, depth, occlusion_probe);
  let reflection = shading_normal.scale( view_vec.dot(&shading_normal)*2.0 ) - view_vec;
  let rcolor = if reflectivity > 0.001 { color_probe(reflection) } else { Vec3::new(0.0,0.0,0.0) };

  let mut ambient;
  //ambient = Vec3::new(0.5f32,0.5f32,0.5f32);

  /*let mut ao = 0f32;
    let rot_to_up = rotate_to_up(n_face);
    const NUM_AO_SAMPLES: uint = 5u;
    for (u,v) in Stratified2dIterator::new(rnd, NUM_AO_SAMPLES, NUM_AO_SAMPLES) {
    let sample_vec = transform(rot_to_up, cosine_hemisphere_sample(u,v) );
  //let sample_vec = cosine_hemisphere_sample(u,v);
    if !occlusion_probe( scale(sample_vec, 0.1f32) ) {
      ao += 1f32/((NUM_AO_SAMPLES*NUM_AO_SAMPLES) as f32);
    }
  };
  ambient = scale(ambient,ao); // todo: add ambient color */

  // Final gather GI
  let gi_normal = if consts::USE_SMOOTH_NORMALS_FOR_GI { n } else { n_face };
  ambient = Vec3::new(0.0,0.0,0.0);
  if depth == 0 && consts::NUM_GI_SAMPLES_SQRT > 0 {
    sample_cosine_hemisphere( rnd, gi_normal, |sample_vec| {
      ambient = ambient + color_probe( sample_vec );
    });
    ambient = ambient.scale( 1.0 / (((consts::NUM_GI_SAMPLES_SQRT * consts::NUM_GI_SAMPLES_SQRT) as f32) * f32::consts::PI ));
  }

  (color * (direct_light + ambient)).lerp(&rcolor, reflectivity)
}


struct Intersection {
  pos: Vec3,
  n: Vec3,
  n_face: Vec3,
  color: Vec3,
  reflectivity: f32
}

#[inline]
fn trace_checkerboard( checkerboard_height: f32, r : &Ray, mint: f32, maxt: f32) -> (Option<Intersection>, f32) {
  // trace against checkerboard first
  let checker_hit_t = (checkerboard_height - r.origin.y) / r.dir.y;

  // compute checkerboard color, if we hit the floor plane
  if checker_hit_t > mint && checker_hit_t < maxt {

    let pos = r.origin + r.dir.scale(checker_hit_t);

    // hacky checkerboard pattern
    let (u,v) = ((pos.x*5.0).floor() as int, (pos.z*5.0).floor() as int);
    let is_white = (u + v) % 2 == 0;
    let color = if is_white { Vec3::new(1.0,1.0,1.0) } else { Vec3::new(1.0,0.5,0.5) };
    let intersection = Some( Intersection{
      pos: pos,
      n: Vec3::new(0.0,1.0,0.0),
      n_face: Vec3::new(0.0,1.0,0.0),
      color: color,
      reflectivity: if is_white {0.3} else {0.0} } );
    (intersection, checker_hit_t)
  } else {
    (None, maxt)
  }
}

#[inline]
fn trace_ray( r : &Ray, mesh : &model::Mesh, mint: f32, maxt: f32) -> Option<Intersection> {

  let use_kd_tree = true;

  let y_size = (mesh.bounding_box.max - mesh.bounding_box.min).y;

  // compute checkerboard color, if we hit the floor plane
  let (checker_intersection, new_maxt) = trace_checkerboard(-y_size*0.5,r,mint,maxt);


  // check scene bounding box first
  if !r.bb_check( new_maxt, mesh.bounding_box ){
    return checker_intersection;
  }

  // trace against scene
  let trace_result = if use_kd_tree {
    trace_kd_tree( &mesh.polys, mesh.kd_tree.nodes, mesh.kd_tree.root, r, r.dir.recip(), mint, new_maxt )
  } else {
    trace_soup( &mesh.polys, r)
  };

  match trace_result {
    Some((hit_info, tri_ix)) if hit_info.t > 0.0 => {
      let pos = r.origin + r.dir.scale(hit_info.t);

      let (i0,i1,i2) = (
        mesh.polys.indices[tri_ix*3  ],
        mesh.polys.indices[tri_ix*3+1],
        mesh.polys.indices[tri_ix*3+2]
      );

      // interpolate vertex normals...
      let n = (
        mesh.polys.normals[i0].scale(hit_info.barycentric.z) +
        mesh.polys.normals[i1].scale(hit_info.barycentric.x) +
        mesh.polys.normals[i2].scale(hit_info.barycentric.y)
      ).normalized();

      // compute face-normal
      let (v0,v1,v2) = (
        mesh.polys.vertices[i0],
        mesh.polys.vertices[i1],
        mesh.polys.vertices[i2]
      );
      let n_face = (v1 - v0).cross(&(v2 - v0)).normalized();

      Some( Intersection{
        pos: pos,
        n: n,
        n_face: n_face,
        color: Vec3::new(1.0,1.0,1.0),
        reflectivity: 0.0 } )
    }
    _ => {
      checker_intersection
    }
  }
}

#[inline]
fn trace_ray_shadow( r: &Ray, mesh: &model::Mesh, mint: f32, maxt: f32) -> bool {

  let y_size = (mesh.bounding_box.max - mesh.bounding_box.min).y;

  // compute checkerboard color, if we hit the floor plane
  let (checker_intersection, new_maxt) = trace_checkerboard(-y_size*0.5,r,mint,maxt);

  if checker_intersection.is_some() {
    return true;
  }

  // check scene bounding box first
  if !r.bb_check( new_maxt, mesh.bounding_box ){
    return false;
  }

  // trace against scene
  trace_kd_tree_shadow( &mesh.polys, mesh.kd_tree.nodes, mesh.kd_tree.root, r, r.dir.recip(), mint, new_maxt )
}


#[inline(always)]
fn get_color( r: &Ray, mesh: &model::Mesh, lights: &[Light], rnd: &RandEnv, tmin: f32, tmax: f32, depth: uint) -> Vec3 {
  let theta = Vec3::new(0.0,1.0,0.0).dot( &r.dir );
  let default_color = Vec3::new(clamp(1.0-theta*4.0,0.0,0.75)+0.25, clamp(0.5-theta*3.0,0.0,0.75)+0.25, theta);    // fake sky colour

  if depth >= consts::MAX_TRACE_DEPTH {
    return default_color;
  }

  match trace_ray( r, mesh, tmin, tmax ) {
    Some(Intersection{pos,n,n_face,color,reflectivity}) => {
      let surface_origin = pos + n_face.scale(0.000002);

      shade(pos, n, n_face, r, color, reflectivity, lights, rnd, depth,
      |occlusion_vec| {
        let occlusion_ray = &Ray{origin: surface_origin, dir: occlusion_vec};
        trace_ray_shadow(occlusion_ray, mesh, 0.0, 1.0)
      },
      |ray_dir| {
        let reflection_ray = &Ray{
          origin: surface_origin,
          dir: ray_dir.normalized(),
        };
        get_color(reflection_ray, mesh, lights, rnd, tmin, tmax, depth + 1)
      })

    }
    _ => { default_color }
  }

}

#[inline]
fn gamma_correct( v : Vec3 ) -> Vec3 {
  Vec3::new(
    v.x.powf( &(1.0/2.2) ),
    v.y.powf( &(1.0/2.2) ),
    v.z.powf( &(1.0/2.2) ),
  )
}

struct TracetaskData {
  meshArc: ::sync::Arc<model::Mesh>,
  horizontalFOV: f32,
  width: uint,
  height: uint,
  sample_grid_size: uint,
  height_start: uint,
  height_stop: uint,
  sample_coverage_inv: f32,
  lights: ~[Light],
  rnd: RandEnv
}

#[inline]
fn tracetask(data: ~TracetaskData) -> ~[Color] {
  match data {
    ~TracetaskData {meshArc: meshArc, horizontalFOV: horizontalFOV, width: width,
    height: height, sample_grid_size: sample_grid_size, height_start: height_start,
    height_stop: height_stop, sample_coverage_inv: sample_coverage_inv, lights: lights,
    rnd: rnd} => {
      let mesh = meshArc.deref();
      let mut img_pixels = Vec::with_capacity(width);
      for row in range( height_start, height_stop ) {
        for column in range( 0, width ) {
          let mut shaded_color = Vec3::new(0.0,0.0,0.0);

          for (u,v) in Stratified2dIterator::new(&rnd, sample_grid_size, sample_grid_size) {
            let sample = match sample_grid_size {
              1 => (0.0,0.0),
              _ => (u-0.5,v-0.5)
            };
            let r = &get_ray(horizontalFOV, width, height, column, row, sample);
            shaded_color = shaded_color + get_color(r, mesh, lights, &rnd, 0.0, f32::INFINITY, 0);
          };
          shaded_color = gamma_correct(shaded_color.scale(sample_coverage_inv * sample_coverage_inv)).scale(255.0);
          let pixel = (
            clamp(shaded_color.x, 0.0, 255.0),
            clamp(shaded_color.y, 0.0, 255.0),
            clamp(shaded_color.z, 0.0, 255.0)).toColor();
          img_pixels.push(pixel)
        }
      }
      img_pixels.move_iter().collect()
    }
  }
}

fn generate_raytraced_image_single(
  mesh: model::Mesh,
  horizontalFOV: f32,
  width: uint,
  height: uint,
  sample_grid_size: uint,
  sample_coverage_inv: f32,
  lights: ~[Light]) -> ~[Color]
{
  let rnd = get_rand_env();
  PixelIterator::new(width,height).map(|pixel| {
    let mut shaded_color = Vec3::new(0.0,0.0,0.0);

    for (u,v) in Stratified2dIterator::new(&rnd, sample_grid_size, sample_grid_size) {
      let sample = match sample_grid_size {
        1 => (0.0,0.0),
        _ => (u-0.5,v-0.5)
      };
      let r = &get_ray(horizontalFOV, width, height, pixel.x, pixel.y, sample );
      shaded_color = shaded_color + get_color(r, &mesh, lights, &rnd, 0.0, f32::INFINITY, 0);
    };
    shaded_color = gamma_correct(shaded_color.scale(sample_coverage_inv*sample_coverage_inv)).scale(255f32);
    (
      clamp(shaded_color.x, 0.0, 255.0),
      clamp(shaded_color.y, 0.0, 255.0),
      clamp(shaded_color.z, 0.0, 255.0)
    ).toColor()
  }).collect()
}

// This fn generates the raytraced image by spawning 'num_tasks' tasks, and letting each
// generate a part of the image. The way the work is divided is not intelligent: it chops
// the image in horizontal chunks of step_size pixels high, and divides these between the
// tasks. There is no work-stealing :(
fn generate_raytraced_image_multi(
  mesh: model::Mesh,
  horizontalFOV: f32,
  width: uint,
  height: uint,
  sample_grid_size: uint,
  sample_coverage_inv: f32,
  lights: ~[Light],
  num_tasks: uint) -> ~[Color]
{
  io::print(format!("using {tasks} tasks ... ", tasks=num_tasks));
  let meshArc = ::sync::Arc::new(mesh);
  let rnd = get_rand_env();
  let mut workers = ~[];
  for _ in range(0,num_tasks) {
    workers.push(concurrent::ConcurrentCalc::new())
  };
  let step_size = 4;
  let mut results = ~[];
  for i in range(0,(height / step_size)+1) {
    let ttd = ~TracetaskData {   // The data required to trace the rays.
      meshArc: meshArc.clone(),
      horizontalFOV: horizontalFOV,
      width: width,
      height: height,
      sample_grid_size: sample_grid_size,
      height_start: min( i * step_size, height),
      height_stop: min( (i + 1) * step_size, height ),
      sample_coverage_inv: sample_coverage_inv,
      lights: lights.clone(),
      rnd: rnd.clone()
    };
    results.push(workers[i % num_tasks].calculate(ttd,tracetask));
  }
  let mut fmap = results.move_iter().flat_map(|f| f.unwrap().move_iter() );
  fmap.collect()
}

pub fn generate_raytraced_image(
  mesh: model::Mesh,
  horizontalFOV: f32,
  width: uint,
  height: uint,
  sample_grid_size: uint) -> ~[Color]
{
  let sample_coverage_inv = 1.0 / (sample_grid_size as f32);
  let lights = ~[  Light::new(Vec3::new(-3.0, 3.0, 0.0),10.0, 0.3, Vec3::new(1.0,1.0,1.0)) ]; //,
  //Light::new(Vec3::new(0f32, 0f32, 0f32), 10f32, 0.25f32, Vec3::new(1f32,1f32,1.0f32))];
  let mut num_tasks = match consts::NUM_THREADS {
    0 => 4u,
    n => n
  };
  if num_tasks > height { num_tasks = height };   // We evaluate complete rows, there is no point in having more tasks than there are rows.
  match num_tasks {
    1 => generate_raytraced_image_single(mesh,horizontalFOV,width,height,sample_grid_size,sample_coverage_inv,lights),
    n => generate_raytraced_image_multi(mesh,horizontalFOV,width,height,sample_grid_size,sample_coverage_inv,lights,n)
  }
}
