#pragma once

#define def_MAX_NUM_OF_THREADS 18

//                 32, 64, 128
#define def_BLOCK_SIZE 64

// MAIN //

//#define def_STATS_PER_THREAD
#define def_DEFAULT_BOUNCE_LIMIT 2
#define def_MAX_BOUNCES 5

#define def_N_FRAMES(64)

#define def_WIDTH (1024)
#define def_HEIGHT (768)
#define def_PIXEL_ARRAY_SIZE (def_WIDTH * def_HEIGHT)

#define def_CAM_POS (d3(def_WIDTH / 2, def_HEIGHT / 2, -10000.0))

#define def_CAM_POS_X (def_WIDTH / 2)
#define def_CAM_POS_Y (def_HEIGHT / 2)
#define def_CAM_POS_Z (-10000.0)

#define def_CAM_INC_X (1.0)
#define def_CAM_INC_Y (1.0)

#ifdef UNIT_FLOAT
#define def_EPSILON (1.19209e-07)
#endif

#ifdef UNIT_DOUBLE
#define def_EPSILON (2.22045e-16)
#endif

#define fatal_error_log ("../rtx_fatal_crach.txt")