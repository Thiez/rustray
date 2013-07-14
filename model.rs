use math3d::*;
use std::unstable::simd::{f32x4};

use std::*;
use std::io::{Reader, ReaderUtil};
use extra::sort;

pub struct polysoup {
    vertices: ~[f32x4],
    indices: ~[uint],
    normals: ~[f32x4]
}

pub struct mesh {
    polys: polysoup,
    kd_tree: kd_tree,
    bounding_box: aabb
}

pub enum axis {
    x,
    y,
    z
}

pub struct kd_tree {
    root: uint,
    nodes: ~[kd_tree_node]
}

pub enum kd_tree_node {
    pub leaf( u32, u32 ),
    pub node( axis, f32, u32 )
}

fn find_split_plane( distances: &[f32], indices: &[uint], faces: &[uint] ) -> f32 {

    let mut face_distances = vec::with_capacity( 3*faces.len() );
    for faces.iter().advance |f| {
        face_distances.push(distances[indices[*f*3u]]);
        face_distances.push(distances[indices[*f*3u+1u]]);
        face_distances.push(distances[indices[*f*3u+2u]]);
    }

    let sorted_distances = sort::merge_sort( face_distances, |a,b| *a<*b );
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

    for faces.iter().advance |f| {
        let f = *f;
        let d0 = distances[indices[f*3u   ]];
        let d1 = distances[indices[f*3u+1u]];
        let d2 = distances[indices[f*3u+2u]];

        let maxdist = d0.max( &d1.max( &d2 ) );
        let mindist = d0.min( &d1.min( &d2 ) );

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

    for faces.iter().advance |f| {
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
    aabbmin: f32x4,
    aabbmax: f32x4,
    indices: &[uint],
    faces: &[uint] ) -> uint {

    if maxdepth == 0u || faces.len() <= 15u {
        return build_leaf( kd_tree_nodes, new_indices, indices, faces );
    }

    let f32x4(ex,ey,ez,_) = aabbmax - aabbmin;
        let axis = if ex > ey && ex > ez {
            x
        } else {
            if ey > ez { y } else { z }
        };

    let dists = match axis { x => xdists, y => ydists, z => zdists };

    let s = find_split_plane( dists, indices, faces );
    let (l,r) = split_triangles( s, dists, indices, faces );

    // Stop when there's too much overlap between the two halves
    if (l.len() + r.len()) as f32 > faces.len() as f32 * 1.5f32 {
        return build_leaf( kd_tree_nodes, new_indices, indices, faces );
    }
    
    let first = f32x4(1.0, 0.0, 0.0, 0.0);
    let second = f32x4(0.0, 1.0, 0.0, 0.0);
    let third = f32x4(0.0, 0.0, 1.0, 0.0);
    let all = f32x4(1.0, 1.0, 1.0, 1.0);
    let ss = f32x4(s, s, s, 0.0);
    // adjust bounding boxes for children
    let sel = match axis {
        x => first,
        y => second,
        z => third,
    };
    let (left_aabbmax,right_aabbmin) = ((ss*sel)+(aabbmax*(all-sel)),(ss*sel)+(aabbmin*(all-sel)));
    
    // allocate node from nodes-array, and recursively build children
    let ix = kd_tree_nodes.len();
    kd_tree_nodes.push( node(axis,0.0,0) );

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
    match nodes[root] {
        node(_,_,r) => {
            let (d0,c0) = count_kd_tree_nodes_( root+1u, nodes);
            let (d1,c1) = count_kd_tree_nodes_( (r as uint), nodes);
            (uint::max(d0,d1)+1u, c0+c1+1u)
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
    let mut faces = vec::with_capacity(max_tri_ix);
    let mut fii = 0u;
    while fii < max_tri_ix {
        faces.push(fii);
        fii += 1u
    }
    let mut aabbmin = f32x4(f32::infinity, f32::infinity, f32::infinity, 0.0);
    let mut aabbmax = f32x4(f32::neg_infinity, f32::neg_infinity, f32::neg_infinity, 0.0);
    for polys.vertices.iter().advance |v| {
        aabbmin = min(*v, aabbmin);
        aabbmax = max(*v, aabbmax);
    }

    let downscale = 1.0 / length(aabbmax - aabbmin);
    let downscale = f32x4(downscale,downscale,downscale,0.0);
    let offset = scale(aabbmin + aabbmax, 0.5);

    let mut transformed_verts = ~[];


    for polys.vertices.iter().advance |v| {
        transformed_verts.push( (*v - offset) * downscale);
    }

    aabbmin = (aabbmin - offset) * downscale;
    aabbmax = (aabbmax - offset) * downscale;

    // de-mux vertices for easier access later
    let mut xdists = ~[];
    let mut ydists = ~[];
    let mut zdists = ~[];

    for transformed_verts.iter().advance |v| {
        let f32x4(vx,vy,vz,_) = *v;
        xdists.push(vx);
        ydists.push(vy);
        zdists.push(vz);
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
                    faces);
    mesh { polys: polysoup{vertices: transformed_verts, indices: new_indices, .. polys},
    kd_tree: kd_tree{ root: rootnode, nodes: nodes} , bounding_box: aabb{min: aabbmin, max: aabbmax} }
}

fn parse_faceindex(s: &str) ->  uint {

    // check for '/', the vertex index is the first
    let ix_str = match s.find('/') {
        Some(slash_ix) => s.slice(0u, slash_ix),
        _ => s
    };
    uint::from_str(ix_str).get()-1u
}

fn read_polysoup(fname: &str) -> polysoup {
    use std::iterator::IteratorUtil;
    let reader = result::get( &io::file_reader( &Path(fname) ) );
    let mut vertices = ~[];
    let mut indices = ~[];

    let mut vert_normals = ~[];

    while !reader.eof() {
        let line : ~str = reader.read_line();
        if line.is_empty() {
            loop;
        }

        let mut num_texcoords = 0u;
        let mut tokens = ~[];
        for line.split_iter(' ').advance |s| { tokens.push(s) };

        if tokens[0] == "v" {
            assert!(tokens.len() == 4u);
            let v = f32x4(  float::from_str(tokens[1]).get() as f32,
                            float::from_str(tokens[2]).get() as f32,
                            float::from_str(tokens[3]).get() as f32,
                            0.0);
            {
                let f32x4(vx,vy,vz,_) = v;
                assert!(vx != f32::NaN);
                assert!(vy != f32::NaN);
                assert!(vz != f32::NaN);
            }

            vertices.push(v);
            vert_normals.push( f32x4(0.0, 0.0, 0.0, 0.0));

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

                for face_triangles.iter().advance |t| {
                    let (i0,i1,i2) = *t;
                    indices.push(i0);
                    indices.push(i1);
                    indices.push(i2);

                    let e1 = vertices[i1] - vertices[i0];
                    let e2 = vertices[i2] - vertices[i0];
                    let n = normalized(cross(e1,e2));

                    vert_normals[i0] = vert_normals[i0] + n;
                    vert_normals[i1] = vert_normals[i1] + n;
                    vert_normals[i2] = vert_normals[i2] + n;
                }
            } else {
                io::println(fmt!("Polygon with %u vertices found. Ignored. Currently rustray only supports 4 vertices", tokens.len() - 1u));
            }
        } else if tokens[0] == "vt" {
            num_texcoords += 1u;
        } else if tokens[0] != "#" {
            io::println(fmt!("Unrecognized line in .obj file: %s", line));
        }

        if num_texcoords > 0u {
            io::println(fmt!("%u texture coordinates ignored", num_texcoords));
        }
    }

    return polysoup{ vertices: vertices,
            indices: indices,
            normals: do vert_normals.map |v|{normalized(*v)} };
}
