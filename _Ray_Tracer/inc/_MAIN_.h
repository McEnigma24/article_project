#pragma once

#include "_PARSER_Format_Buffer_Common_.h"

#define NUM_OF_CATEGORIES 13
//              1    2    3    4                                           8    9    10    11 12
//        PROCESSOR MODEL ARCH UNIT                                    SCENE LIGHTS SPHERES BOUNCES NUM_OF_THREADS
//                                        5             6             7 13
//                                  PHYSICAL_CORES LOGICAL_CORES CORE_PROPORTION RATING
#define format                                                                                                         \
    "%s - %s - %s - %s -> Core Details %s %s %s -> Scene Details %s - %s - %s - %s -> Execution Details - 0 th %s / "  \
    "18 - %s - <%llu, %llu, %llu> Odchylenie standardowe: %lf",

enum category_index
{
    PROCESSOR,
    MODEL,
    ARCH,
    UNIT,
    PHYSICAL_CORES,
    LOGICAL_CORES,
    CORE_PROPORTION,
    SCENE,
    LIGHTS,
    SPHERES,
    BOUNCES,
    NUM_OF_THREADS,
    RATING
};

#define GET(t, index) get<index>(t)

#define tupl                                                                                                           \
    tuple<double, u64, u64, u64, string, string, string, string, string, string, string, string, string, string,       \
          string, string, string>

#define add_to_all(a, b)                                                                                               \
                                                                                                                       \
    add(a, processor, b), add(a, model, b), add(a, arch, b), add(a, unit, b), add(a, physical_cores, b),               \
        add(a, logical_cores, b), add(a, core_proportion, b), add(a, scene, b), add(a, lights, b), add(a, spheres, b), \
        add(a, bounces, b), add(a, num_of_threads, b), add(a, rating, b)

#define adding_buffer_init                                                                                             \
                                                                                                                       \
    add_array(processor);                                                                                              \
    add_array(model);                                                                                                  \
    add_array(arch);                                                                                                   \
    add_array(unit);                                                                                                   \
    add_array(physical_cores);                                                                                         \
    add_array(logical_cores);                                                                                          \
    add_array(core_proportion);                                                                                        \
    add_array(scene);                                                                                                  \
    add_array(lights);                                                                                                 \
    add_array(spheres);                                                                                                \
    add_array(bounces);                                                                                                \
    add_array(num_of_threads);                                                                                         \
    add_array(rating);

#define category_assignment                                                                                            \
    add_unique(PROCESSOR, add(read_, processor, ));                                                                    \
    add_unique(MODEL, add(read_, model, ));                                                                            \
    add_unique(ARCH, add(read_, arch, ));                                                                              \
    add_unique(UNIT, add(read_, unit, ));                                                                              \
    add_unique(PHYSICAL_CORES, add(read_, physical_cores, ));                                                          \
    add_unique(LOGICAL_CORES, add(read_, logical_cores, ));                                                            \
    add_unique(CORE_PROPORTION, add(read_, core_proportion, ));                                                        \
    add_unique(SCENE, add(read_, scene, ));                                                                            \
    add_unique(LIGHTS, add(read_, lights, ));                                                                          \
    add_unique(SPHERES, add(read_, spheres, ));                                                                        \
    add_unique(BOUNCES, add(read_, bounces, ));                                                                        \
    add_unique(NUM_OF_THREADS, add(read_, num_of_threads, ));                                                          \
    add_unique(RATING, add(read_, rating, ));

#define input_tupl_variables                                                                                           \
    get<4>(variables).c_str(), get<5>(variables).c_str(), get<6>(variables).c_str(), get<7>(variables).c_str(),        \
        get<8>(variables).c_str(), get<9>(variables).c_str(), get<10>(variables).c_str(), get<11>(variables).c_str(),  \
        get<12>(variables).c_str(), get<13>(variables).c_str(), get<14>(variables).c_str(),                            \
        get<15>(variables).c_str(), get<16>(variables).c_str()

#define condition                                                                                                      \
    (param == add(read_, processor, )) || (param == add(read_, model, )) || (param == add(read_, arch, )) ||           \
        (param == add(read_, unit, )) || (param == add(read_, physical_cores, )) ||                                    \
        (param == add(read_, logical_cores, )) || (param == add(read_, core_proportion, )) ||                          \
        (param == add(read_, scene, )) || (param == add(read_, lights, )) || (param == add(read_, spheres, )) ||       \
        (param == add(read_, bounces, )) || (param == add(read_, num_of_threads, )) || (param == add(read_, rating, ))
