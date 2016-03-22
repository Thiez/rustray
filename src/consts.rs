// TODO: These things should all be overridable by the command line
pub static WIDTH: u32 = 1024;
pub static HEIGHT: u32 = 1024;
pub static FOV: f32 = 3.14159f32 / 3f32;
pub static SAMPLE_GRID_SIZE: u32 = 1;
pub static NUM_GI_SAMPLES_SQRT: u32 = 4;
pub static NUM_LIGHT_SAMPLES: u32 = 8;
pub static MAX_TRACE_DEPTH: u32 = 1;
pub static USE_SMOOTH_NORMALS_FOR_GI: bool = true;
pub static USE_SMOOTH_NORMALS_FOR_DIRECT_LIGHTING: bool = true;
pub static NUM_THREADS: u32 = 16;
