use super::math3d::{Vec3,BoundingBox};

use std::f32;
use std::fs::{OpenOptions};
use std::io::{BufReader,BufRead};
use std::path::{Path};

pub struct Polysoup {
  pub vertices: Vec<Vec3>,
  pub indices: Vec<u32>,
  pub normals: Vec<Vec3>,
}

pub struct Mesh {
  pub polys: Polysoup,
  pub kd_tree: KdTree,
  pub bounding_box: BoundingBox,
}

#[derive(Clone,Copy)]
pub enum Axis {
  AxisX,
  AxisY,
  AxisZ,
}

pub struct KdTree {
  pub root: u32,
  pub nodes: Vec<KdTreeNode>,
}

#[derive(Clone,Copy)]
pub enum KdTreeNode {
  KdLeaf( u32, u32 ),
  KdNode( Axis, f32, u32 )
}

fn find_split_plane( distances: &[f32], indices: &[u32], faces: &[u32] ) -> f32 {
  use std::cmp::Ordering::{Less,Equal,Greater};
  let mut face_distances = Vec::with_capacity( 3*faces.len() );
  for f in faces.iter().map(|&f|(f*3) as usize) {
    face_distances.push(distances[indices[f+0] as usize]);
    face_distances.push(distances[indices[f+1] as usize]);
    face_distances.push(distances[indices[f+2] as usize]);
  }
  face_distances.sort_by(|&a,&b| if a<b { Less } else if a == b { Equal } else { Greater });

  let sorted_distances = &face_distances;

  let n = sorted_distances.len();
  if n % 2 == 0 {
    sorted_distances[ n/2 ]
  } else {
    (sorted_distances[ n/2 -1] + sorted_distances[ n/2 ]) * 0.5f32
  }
}

fn split_triangles(splitter: f32, distances: &[f32], indices: &[u32], faces: &[u32] ) -> (Vec<u32>,Vec<u32>) {
  let (mut l, mut r) = (vec![], vec![]);

  for &f in faces.iter() {
    let fi = (f*3) as usize;
    let d0 = distances[indices[fi+0] as usize];
    let d1 = distances[indices[fi+1] as usize];
    let d2 = distances[indices[fi+2] as usize];

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
  kd_tree_nodes : &mut Vec<KdTreeNode>,
  new_indices: &mut Vec<u32>,
  indices: &[u32],
  faces: &[u32]
  ) -> u32 {

  let next_face_ix = (new_indices.len() as u32) / 3;
  kd_tree_nodes.push(KdTreeNode::KdLeaf( next_face_ix, (faces.len() as u32) ));

  for f in faces.iter().map(|&f|(f*3) as usize) {
    new_indices.push(indices[f+0]);
    new_indices.push(indices[f+1]);
    new_indices.push(indices[f+2]);
  }
  (kd_tree_nodes.len() - 1) as u32
}

fn build_kd_tree<'r>(
  kd_tree_nodes : &mut Vec<KdTreeNode>,
  new_indices: &mut Vec<u32>,
  maxdepth: u32,
  xdists: &'r [f32],
  ydists: &'r [f32],
  zdists: &'r [f32],
  aabbmin: Vec3,
  aabbmax: Vec3,
  indices: &[u32],
  faces: &[u32] ) -> u32 {

  if maxdepth == 0 || faces.len() <= 15 {
    return build_leaf( kd_tree_nodes, new_indices, indices, faces );
  }

  let extent = aabbmax - aabbmin;
  let axis = if extent.x > extent.y && extent.x > extent.z {
    Axis::AxisX
  } else {
    if extent.y > extent.z { Axis::AxisY } else { Axis::AxisZ }
  };

  let dists = match axis { Axis::AxisX => xdists, Axis::AxisY => ydists, Axis::AxisZ => zdists };

  let s = find_split_plane( dists, indices, faces );
  let (l,r) = split_triangles( s, dists, indices, faces );

  // Stop when there's too much overlap between the two halves
  if (l.len() + r.len()) as f32 > faces.len() as f32 * 1.5f32 {
    return build_leaf( kd_tree_nodes, new_indices, indices, faces );
  }

  // adjust bounding boxes for children
  let (left_aabbmax,right_aabbmin) = match axis {
    Axis::AxisX => (Vec3{x:s, ..aabbmax},Vec3{x:s, ..aabbmin}),
    Axis::AxisY => (Vec3{y:s, ..aabbmax},Vec3{y:s, ..aabbmin}),
    Axis::AxisZ => (Vec3{z:s, ..aabbmax},Vec3{z:s, ..aabbmin}),
  };

  // allocate node from nodes-array, and recursively build children
  let ix = kd_tree_nodes.len();
  kd_tree_nodes.push( KdTreeNode::KdNode(axis,0f32,0u32) );

  build_kd_tree(
    &mut *kd_tree_nodes,
    &mut *new_indices,
    maxdepth - 1,
    xdists,
    ydists,
    zdists,
    aabbmin,
    left_aabbmax,
    indices,
    &l);
  // left child ix is implied to be ix+1

  let right_child_ix = build_kd_tree(
    &mut *kd_tree_nodes,
    &mut *new_indices,
    maxdepth - 1,
    xdists,
    ydists,
    zdists,
    right_aabbmin,
    aabbmax,
    indices,
    &r);

  kd_tree_nodes[ix] = KdTreeNode::KdNode(axis, s as f32, right_child_ix as u32);

  ix as u32
}

pub fn count_kd_tree_nodes( t: &KdTree ) -> (u32, u32) {
  count_kd_tree_nodes_( t.root, &t.nodes )
}

fn count_kd_tree_nodes_( root: u32, nodes: &[KdTreeNode]) -> (u32, u32) {
  use std::cmp::max;
  match nodes[root as usize] {
    KdTreeNode::KdNode(_,_,r) => {
      let (d0,c0) = count_kd_tree_nodes_( root+1, nodes);
      let (d1,c1) = count_kd_tree_nodes_( r, nodes);
      (max(d0,d1)+1, c0+c1+1)
    }
    KdTreeNode::KdLeaf(_,_) => (1, 1)
  }
}

pub fn read_mesh(fname: &str) -> Mesh {
  print!("Reading model file...");
  let polys = read_polysoup( fname );

  print!("Building kd-tree... ");

  // just create a vector of 0..N-1 as our face array
  let max_tri_ix = (polys.indices.len()/3 - 1) as u32;
  let mut faces = Vec::with_capacity(max_tri_ix as usize);
  let mut fii = 0;
  while fii < max_tri_ix {
    faces.push(fii);
    fii += 1
  }
  let mut aabbmin = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
  let mut aabbmax = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
  for v in polys.vertices.iter() {
    aabbmin = v.min(&aabbmin);
    aabbmax = v.max(&aabbmax);
  }

  let downscale = 1.0f32 / (aabbmax - aabbmin).length();
  let offset = (aabbmin + aabbmax).scale(0.5f32);

  let mut transformed_verts = Vec::new();


  for &v in polys.vertices.iter() {
    transformed_verts.push((v - offset).scale(downscale));
  }

  aabbmin = (aabbmin - offset).scale(downscale);
  aabbmax = (aabbmax - offset).scale(downscale);

  // de-mux vertices for easier access later
  let mut xdists = vec![];
  let mut ydists = vec![];
  let mut zdists = vec![];

  for v in transformed_verts.iter() {
    xdists.push(v.x);
    ydists.push(v.y);
    zdists.push(v.z);
  }

  let mut nodes = vec![];
  let mut new_indices = vec![];

  let rootnode = build_kd_tree(
    &mut nodes,
    &mut new_indices,
    100,
    &xdists,
    &ydists,
    &zdists,
    aabbmin,
    aabbmax,
    &polys.indices,
    &faces);
  Mesh { polys: Polysoup{vertices: transformed_verts, indices: new_indices, .. polys},
  kd_tree: KdTree{ root: rootnode, nodes: nodes} , bounding_box: BoundingBox{min: aabbmin, max: aabbmax} }
}

fn parse_faceindex(s: &str) ->  u32 {
  // check for '/', the vertex index is the first
  s.find('/')
      .map(|ix|&s[0..ix])
      .or(Some(s))
      .map(str::parse::<u32>)
      .and_then(Result::ok)
      .unwrap()-1
}

fn read_polysoup(fname: &str) -> Polysoup {
  let reader = OpenOptions::new()
      .read(true)
      .open(&Path::new(fname))
      .map(BufReader::new)
      .ok()
      .expect("Could not open file!");
  let mut vertices = vec![];
  let mut indices = vec![];

  let mut vert_normals : Vec<Vec3> = vec![];

  for line in reader
      .lines()
      .filter_map(Result::ok)
      .filter(|s|!s.is_empty()) {

    let mut num_texcoords = 0;
    let tokens = line.split(' ').map(str::trim).collect::<Vec<_>>();

    if tokens[0] == "v" {
      assert!(tokens.len() == 4);

      let v = Vec3::new(
        tokens[1].parse().unwrap(),
        tokens[2].parse().unwrap(),
        tokens[3].parse().unwrap()
      );

      assert!(!v.x.is_nan());
      assert!(!v.y.is_nan());
      assert!(!v.z.is_nan());

      vertices.push(v);
      vert_normals.push(Vec3::new(0f32,0f32,0f32));

    } else if tokens[0] == "f" {
      if tokens.len() == 4 || tokens.len() == 5 {
        let mut face_triangles = vec![];

        if tokens.len() == 4 {
          let (i0,i1,i2) = (
            parse_faceindex(tokens[1]),
            parse_faceindex(tokens[2]),
            parse_faceindex(tokens[3])
          );

          face_triangles.push((i0, i1, i2));
        } else {
          assert!(tokens.len() == 5);
          // quad, triangulate
          let (i0,i1,i2,i3) = (
            parse_faceindex(tokens[1]),
            parse_faceindex(tokens[2]),
            parse_faceindex(tokens[3]),
            parse_faceindex(tokens[4])
          );

          face_triangles.push((i0,i1,i2));
          face_triangles.push((i0,i2,i3));
        }

        for &(i0,i1,i2) in face_triangles.iter() {
          indices.push(i0);
          indices.push(i1);
          indices.push(i2);

          let (i0,i1,i2) = (i0 as usize, i1 as usize, i2 as usize);

          let e1 = vertices[i1] - vertices[i0];
          let e2 = vertices[i2] - vertices[i0];
          let n = (e1.cross(&e2)).normalized();

          vert_normals[i0] = vert_normals[i0] + n;
          vert_normals[i1] = vert_normals[i1] + n;
          vert_normals[i2] = vert_normals[i2] + n;
        }
      } else {
        println!("Polygon with {} vertices found. Ignored. Currently rustray only supports 4 vertices", tokens.len() - 1);
      }
    } else if tokens[0] == "vt" {
      num_texcoords += 1;
    } else if tokens[0] != "#" {
      println!("Unrecognized line in .obj file: {}", line);
    }

    if num_texcoords > 0 {
      println!("{} texture coordinates ignored", num_texcoords);
    }
  }

  Polysoup{
    vertices: vertices,
    indices: indices,
    normals: vert_normals.into_iter().map( |vec|vec.normalized() ).collect()
  }
}
