#![crate_name = "rustray"]

#![crate_type = "bin"]
#![feature(unboxed_closures, fnbox)]

extern crate time;
extern crate rand;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

pub mod consts;
pub mod math3d;
pub mod model;
pub mod raytracer;
pub mod concurrent;

fn write_ppm( fname: &str, width: u32, height: u32, pixels: &[raytracer::Color] ){
  let mut writer = OpenOptions::new()
      .write(true)
      .create(true)
      .open(&Path::new(fname)).map(BufWriter::new).unwrap();
  let _ = writer.write_all( format!("P6\n{} {}\n255\n", width, height).as_bytes() );
  for pixel in pixels.iter() {
    let _ = writer.write(&[pixel.r, pixel.g, pixel.b][..]);
  };
}

fn main()
{
  // Get command line args
  let args = std::env::args().collect::<Vec<_>>();

  if args.len() != 2 {
    println!("Usage: rustray OBJ");
    println!("");
    println!("For example:");
    println!("   $ wget http://groups.csail.mit.edu/graphics/classes/6.837/F03/models/cow-nonormals.obj");
    println!("   $ ./rustray cow-nonormals.obj");
    println!("   $ gimp oput.ppm");
    println!("");
    panic!();
  }

  let start = ::time::precise_time_s();


  println!("Reading {}...", args[1]);
  let model = model::read_mesh( &args[1] );

  let (depth,count) = model::count_kd_tree_nodes( &model.kd_tree );

  println!("Done.");
  println!("Loaded model.");
  println!("\tVerts: {}, Tris: {}",model.polys.vertices.len(),model.polys.indices.len()/3);
  println!("\tKD-tree depth: {}, nodes: {}", depth, count);

  print!("Tracing rays... ");
  let start_tracing = ::time::precise_time_s();
  let pixels = raytracer::generate_raytraced_image(model, consts::FOV, consts::WIDTH, consts::HEIGHT, consts::SAMPLE_GRID_SIZE);
  println!("Done!");
  let end_tracing = ::time::precise_time_s();

  let outputfile = "./oput.ppm";
  println!("Writing {}...", outputfile);
  write_ppm( outputfile, consts::WIDTH, consts::HEIGHT, &pixels );
  println!("Done!");

  let end = ::time::precise_time_s();
  println!("Total time: {}s, of which tracing: {}", (end - start), (end_tracing - start_tracing));
}
