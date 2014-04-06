use super::math3d::{vec3,mtx33,dot,lerp,scale,length_sq,length,normalized,recip,mul,add,sub,cross,min,max,Ray,Triangle,HitResult,aabb,cosine_hemisphere_sample,rotate_to_up,rotate_y,transform,mul_mtx33,transposed};

use std::f32;
use std::io;
use std::io::{Reader,BufferedReader,File,Open,Read};
use std::path;

pub struct polysoup {
  pub vertices: ~[vec3],
  pub indices: ~[uint],
  pub normals: ~[vec3],
}

pub struct mesh {
  pub polys: polysoup,
  pub kd_tree: kd_tree,
  pub bounding_box: aabb,
}

pub enum axis {
  x,
  y,
  z
}

pub struct kd_tree {
  pub root: uint,
  pub nodes: ~[kd_tree_node]
}

pub enum kd_tree_node {
  leaf( u32, u32 ),
  node( axis, f32, u32 )
}

fn find_split_plane( distances: &[f32], indices: &[uint], faces: &[uint] ) -> f32 {
  use std::cmp::{Less,Equal,Greater};
  let mut face_distances = Vec::with_capacity( 3*faces.len() );
  for f in faces.iter() {
    face_distances.push(distances[indices[*f*3u]]);
    face_distances.push(distances[indices[*f*3u+1u]]);
    face_distances.push(distances[indices[*f*3u+2u]]);
  }
  face_distances.sort_by(|&a,&b| if a<b { Less } else if a == b { Equal } else { Greater });

  let sorted_distances = face_distances.as_slice();

  let n = sorted_distances.len();
  if n % 2u == 0u {
    sorted_distances[ n/2u ]
  } else {
    (sorted_distances[ n/2u -1u] + sorted_distances[ n/2u ]) * 0.5f32
  }
}

fn split_triangles( splitter: f32, distances: &[f32], indices: &[uint], faces: &[uint] ) -> (~[uint],~[uint]) {
  let mut l = ~[];
  let mut r = ~[];

  for f in faces.iter() {
    let f = *f;
    let d0 = distances[indices[f*3u   ]];
    let d1 = distances[indices[f*3u+1u]];
    let d2 = distances[indices[f*3u+2u]];

    let maxdist = d0.max( d1.max( d2 ) );
    let mindist = d0.min( d1.min( d2 ) );

    if mindist <= splitter {
      l.push(f);
    }

    if maxdist >= splitter {
      r.push(f);
    }
  }

  (l, r)
}

fn build_leaf(
  kd_tree_nodes : &mut ~[kd_tree_node],
  new_indices: &mut ~[uint],
  indices: &[uint],
  faces: &[uint]
  ) -> uint {

  let next_face_ix : u32 = (new_indices.len() as u32) / 3u32;
  kd_tree_nodes.push(leaf( next_face_ix, (faces.len() as u32) ));

  for f in faces.iter() {
    let f = *f;
    new_indices.push( indices[f*3]   );
    new_indices.push( indices[f*3+1] );
    new_indices.push( indices[f*3+2] );
  }
  kd_tree_nodes.len() - 1u
}

fn build_kd_tree<'r>(
  kd_tree_nodes : &mut ~[kd_tree_node],
  new_indices: &mut ~[uint],
  maxdepth: uint,
  xdists: &'r [f32],
  ydists: &'r [f32],
  zdists: &'r [f32],
  aabbmin: vec3,
  aabbmax: vec3,
  indices: &[uint],
  faces: &[uint] ) -> uint {

  if maxdepth == 0u || faces.len() <= 15u {
    return build_leaf( kd_tree_nodes, new_indices, indices, faces );
  }

  let extent = sub(aabbmax, aabbmin);
  let axis = if extent.x > extent.y && extent.x > extent.z {
    x
  } else {
    if extent.y > extent.z { y } else { z }
  };

  let dists = match axis { x => xdists, y => ydists, z => zdists };

  let s = find_split_plane( dists, indices, faces );
  let (l,r) = split_triangles( s, dists, indices, faces );

  // Stop when there's too much overlap between the two halves
  if (l.len() + r.len()) as f32 > faces.len() as f32 * 1.5f32 {
    return build_leaf( kd_tree_nodes, new_indices, indices, faces );
  }

  // adjust bounding boxes for children
  let (left_aabbmax,right_aabbmin) = match axis {
    x => (vec3{x:s, ..aabbmax},vec3{x:s, ..aabbmin}),
    y => (vec3{y:s, ..aabbmax},vec3{y:s, ..aabbmin}),
    z => (vec3{z:s, ..aabbmax},vec3{z:s, ..aabbmin}),
  };

  // allocate node from nodes-array, and recursively build children
  let ix = kd_tree_nodes.len();
  kd_tree_nodes.push( node(axis,0f32,0u32) );

  build_kd_tree(
    &mut *kd_tree_nodes,
    &mut *new_indices,
    maxdepth - 1u,
    xdists,
    ydists,
    zdists,
    aabbmin,
    left_aabbmax,
    indices,
    l
    );
  // left child ix is implied to be ix+1

  let right_child_ix = build_kd_tree(
    &mut *kd_tree_nodes,
    &mut *new_indices,
    maxdepth - 1u,
    xdists,
    ydists,
    zdists,
    right_aabbmin,
    aabbmax,
    indices,
    r
    );

  kd_tree_nodes[ix] = node(axis, s as f32, right_child_ix as u32);

  return ix;
}

pub fn count_kd_tree_nodes( t: &kd_tree ) -> (uint, uint) {
  count_kd_tree_nodes_( t.root, t.nodes )
}

fn count_kd_tree_nodes_( root: uint, nodes: &[kd_tree_node]) -> (uint, uint) {
  use std::cmp::max;
  match nodes[root] {
    node(_,_,r) => {
      let (d0,c0) = count_kd_tree_nodes_( root+1u, nodes);
      let (d1,c1) = count_kd_tree_nodes_( (r as uint), nodes);
      (max(d0,d1)+1u, c0+c1+1u)
    }
    leaf(_,_) => (1u, 1u)
  }
}

pub fn read_mesh(fname: &str) -> mesh {
  io::print("Reading model file...");
  let polys = read_polysoup( fname );

  io::print("Building kd-tree... ");

  // just create a vector of 0..N-1 as our face array
  let max_tri_ix = polys.indices.len()/3u -1u;
  let mut faces = Vec::with_capacity(max_tri_ix);
  let mut fii = 0u;
  while fii < max_tri_ix {
    faces.push(fii);
    fii += 1u
  }
  let mut aabbmin = vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY);
  let mut aabbmax = vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
  for v in polys.vertices.iter() {
    aabbmin = min(*v, aabbmin);
    aabbmax = max(*v, aabbmax);
  }

  let downscale = 1.0f32 / length(sub(aabbmax,aabbmin));
  let offset = scale(add(aabbmin, aabbmax), 0.5f32);

  let mut transformed_verts = ~[];


  for v in polys.vertices.iter() {
    transformed_verts.push(scale(sub(*v, offset), downscale));
  }

  aabbmin = scale(sub(aabbmin, offset), downscale);
  aabbmax = scale(sub(aabbmax, offset), downscale);

  // de-mux vertices for easier access later
  let mut xdists = ~[];
  let mut ydists = ~[];
  let mut zdists = ~[];

  for v in transformed_verts.iter() {
    xdists.push(v.x);
    ydists.push(v.y);
    zdists.push(v.z);
  }

  let mut nodes = ~[];
  let mut new_indices = ~[];

  let rootnode = build_kd_tree(
    &mut nodes,
    &mut new_indices,
    100u,
    xdists,
    ydists,
    zdists,
    aabbmin,
    aabbmax,
    polys.indices,
    faces.as_slice());
  mesh { polys: polysoup{vertices: transformed_verts, indices: new_indices, .. polys},
  kd_tree: kd_tree{ root: rootnode, nodes: nodes} , bounding_box: aabb{min: aabbmin, max: aabbmax} }
}

fn parse_faceindex(s: &str) ->  uint {

  // check for '/', the vertex index is the first
  let ix_str: &str = match s.find('/') {
    Some(slash_ix) => s.slice(0u, slash_ix),
    _ => s
  };
  from_str::<uint>(ix_str).unwrap()-1u
}

fn read_polysoup(fname: &str) -> polysoup {
  let mut reader = File::open_mode( &Path::new(fname), Open, Read ).map(|f| BufferedReader::new(f)).ok().expect("Could not open file!");
  let mut vertices = ~[];
  let mut indices = ~[];

  let mut vert_normals : ~[vec3] = ~[];

  loop {
    let line = match reader.read_line() {
      Ok(line) => line,
      Error => break,
    };
    if line.is_empty() {
      continue;
    }

    let mut num_texcoords = 0u;
    let mut tokens = ~[];
    for s in line.split(' ') { tokens.push(s.trim()) };

    if tokens[0] == "v" {
      assert!(tokens.len() == 4u);

      let v = vec3(
        from_str::<f32>(tokens[1]).unwrap(),
        from_str::<f32>(tokens[2]).unwrap(),
        from_str::<f32>(tokens[3]).unwrap()
      );

      assert!(!v.x.is_nan());
      assert!(!v.y.is_nan());
      assert!(!v.z.is_nan());

      vertices.push(v);
      vert_normals.push(vec3(0f32,0f32,0f32));

    } else if tokens[0] == "f" {
      if tokens.len() == 4u || tokens.len() == 5u {
        let mut face_triangles = ~[];

        if tokens.len() == 4u {
          let (i0,i1,i2) = (  parse_faceindex(tokens[1]),
          parse_faceindex(tokens[2]),
          parse_faceindex(tokens[3]) );

          face_triangles.push((i0, i1, i2));
        } else {
          assert!(tokens.len() == 5u);
          // quad, triangulate
          let (i0,i1,i2,i3) = (   parse_faceindex(tokens[1]),
          parse_faceindex(tokens[2]),
          parse_faceindex(tokens[3]),
          parse_faceindex(tokens[4]) );

          face_triangles.push((i0,i1,i2));
          face_triangles.push((i0,i2,i3));
        }

        for t in face_triangles.iter() {
          let (i0,i1,i2) = *t;
          indices.push(i0);
          indices.push(i1);
          indices.push(i2);

          let e1 = sub(vertices[i1], vertices[i0]);
          let e2 = sub(vertices[i2], vertices[i0]);
          let n = normalized(cross(e1,e2));

          vert_normals[i0] = add( vert_normals[i0], n );
          vert_normals[i1] = add( vert_normals[i1], n );
          vert_normals[i2] = add( vert_normals[i2], n );
        }
      } else {
        io::println(format!("Polygon with {verts} vertices found. Ignored. Currently rustray only supports 4 vertices", verts=(tokens.len() - 1u)));
      }
    } else if tokens[0] == "vt" {
      num_texcoords += 1u;
    } else if tokens[0] != "#" {
      io::println(format!("Unrecognized line in .obj file: {s}", s=line));
    }

    if num_texcoords > 0u {
      io::println(format!("{num} texture coordinates ignored", num=num_texcoords));
    }
  }
  println!("Foobar");

  polysoup{
    vertices: vertices,
    indices: indices,
    normals: vert_normals.move_iter().map(normalized).collect()
  }
}
