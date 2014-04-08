#![crate_id = "rustray#0.11"]

#![comment = "A toy ray tracer in Rust"]
#![author = "Sebastian Sylvan"]
#![license = "Unknown"]
#![crate_type = "bin"]

extern crate time;
extern crate sync;
extern crate rand;

use std::{io,os};
use std::io::{BufferedWriter,Writer,File,Open,Write};
use std::path::Path;

pub mod consts;
pub mod math3d;
pub mod model;
pub mod raytracer;
pub mod concurrent;

fn write_ppm( fname: &str, width: uint, height: uint, pixels: &[raytracer::Color] ){
  let mut writer = File::open_mode( &Path::new(fname), Open, Write ).map(|f|BufferedWriter::new(f)).unwrap();
  let _ = writer.write_str( format!("P6\n{} {}\n255\n", width, height) );
  for pixel in pixels.iter() {
    let _ = writer.write([pixel.r, pixel.g, pixel.b]);
  };
}

fn main()
{
  // Get command line args
  let args = os::args();

  if args.len() != 2u {
    io::println("Usage: rustray OBJ");
    io::println("");
    io::println("For example:");
    io::println("   $ wget http://groups.csail.mit.edu/graphics/classes/6.837/F03/models/cow-nonormals.obj");
    io::println("   $ ./rustray cow-nonormals.obj");
    io::println("   $ gimp oput.ppm");
    io::println("");
    fail!();
  }

  let start = ::time::precise_time_s();


  io::println(~"Reading \"" + args[1] + "\"...");
  let model = model::read_mesh( args[1] );

  let (depth,count) = model::count_kd_tree_nodes( &model.kd_tree );

  io::println(format!("Done.\nLoaded model.\n\tVerts: {verts}, Tris: {tris}\n\tKD-tree depth: {depth}, nodes: {nodes}",
  verts=model.polys.vertices.len(),
  tris=model.polys.indices.len()/3u,
  depth=depth, nodes=count));

  io::print("Tracing rays... ");
  let start_tracing = ::time::precise_time_s();
  let pixels = raytracer::generate_raytraced_image(model, consts::FOV, consts::WIDTH, consts::HEIGHT, consts::SAMPLE_GRID_SIZE);
  io::println("Done!");
  let end_tracing = ::time::precise_time_s();

  let outputfile = "./oput.ppm";
  io::print(~"Writing \"" + outputfile + "\"...");
  write_ppm( outputfile, consts::WIDTH, consts::HEIGHT, pixels );
  io::println("Done!");

  let end = ::time::precise_time_s();
  io::print(format!("Total time: {total}s, of which tracing: {tracing}\n", total= (end - start), tracing=(end_tracing - start_tracing)) );
}
