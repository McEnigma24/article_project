#include "Multithreaded_Memcpy.h"
#include "Nano_Timer.h"
#include "Randoms.h"
#include "visualizer.h"
#include <cmath>
#include <omp.h>

#define M_PI (3.14159265358979323846)

#if defined(CPU) && defined(D_BUFF_SAME_OBJ)
class Sim_sphere
{
    d3 position[2];
    sim_unit r[2];
    sim_unit T[2];

public:
    Sim_sphere() { memset(this, 0, sizeof(*this)); }

    d3& get_position(u8 index) { return position[index]; }
    sim_unit& get_r(u8 index) { return r[index]; }
    sim_unit& get_T(u8 index) { return T[index]; }

    void init(const d3& _position, const sim_unit& _r, const sim_unit& _T)
    {
        position[0] = _position;
        // position[1] = _position;
        r[0] = _r;
        // r[1] = _r;
        T[0] = _T;
        // T[1] = _T;
    }
};
#else
class Sim_sphere
{
    d3 position;
    sim_unit r;
    sim_unit T;

public:
    Sim_sphere() { memset(this, 0, sizeof(*this)); }

    d3& get_position() { return position; }
    sim_unit& get_r() { return r; }
    sim_unit& get_T() { return T; }

    void init(const d3& _position, const sim_unit& _r, const sim_unit& _T)
    {
        position = _position;
        r = _r;
        T = _T;
    }
};
#endif

// na początku można zrobić z tymi prymitywnymi RÓWNOLEGŁYMI //
// temp_dist                -> dla wszystkich par (nowe pary, jeśli ilość sfer się zmieni)
//                              ograniczone o jakiś dystans
// collision_resolution      -> wszystkie, i poprawki tylko dla tych co się nachodzą

// #define BOUND_CHECKS_V3(...) __VA_ARGS__
#define BOUND_CHECKS_V3(...)

template <typename T>
class v3
{
    u64 WIDTH;
    u64 HEIGHT;
    u64 DEPTH;
    vector<T> buffer;

    bool bound_checks(u64 x, u64 y, u64 z) { return (0 <= x && x < WIDTH) && (0 <= y && y < HEIGHT) && (0 <= z && z < DEPTH); }
    u64 d3_to_d1(u64 x, u64 y, u64 z) const { return x + y * (WIDTH) + z * (WIDTH * HEIGHT); }

public:
    v3() : WIDTH(0), HEIGHT(0), DEPTH(0) {}

    void set_sizes(size_t width, size_t height, size_t depth)
    {
        WIDTH = width;
        HEIGHT = height;
        DEPTH = depth;

        buffer.resize(WIDTH * HEIGHT * DEPTH);
    }

    u64 get_width() const { return WIDTH; }
    u64 get_height() const { return HEIGHT; }
    u64 get_depth() const { return DEPTH; }
    u64 get_total_number() const { return buffer.size(); }
    T* data() { return buffer.data(); }

    T* get(u64 x, u64 y, u64 z)
    {
        BOUND_CHECKS_V3(if (!bound_checks(x, y, z)) FATAL_ERROR("accesing beyond vector bounds");)

        return &buffer[d3_to_d1(x, y, z)];
    }

    T* get(u64 i)
    {
        BOUND_CHECKS_V3(if (!(i < buffer.size())) FATAL_ERROR("accesing beyond vector bounds");)

        return &buffer[i];
    }
};

class Memory_index_cyclic
{
    u64 index;
    const u64 cyclic_length;

public:
    Memory_index_cyclic(const u64 _cyclic_length) : index(0), cyclic_length(_cyclic_length) {}

    u64 get() const { return index; }
    u64 get_next() const { return (index + 1) % cyclic_length; }

    void switch_to_next() { index = get_next(); }
};

class Computation_Box
{
    sim_unit SIM_scale;
    sim_unit SIM_initial_sphere_separation;
    sim_unit SIM_initial_radious;
    sim_unit SIM_initial_temperature;

    sim_unit SIM_density;
    sim_unit SIM_heat_capacity;    // ciepło właściwe
    sim_unit SIM_wall_heat_reach;  // odległość od wybranej ściany, do której działa ogrzewanie
    sim_unit SIM_wall_heat_value;  // ilość ogrzewania od ściany
    sim_unit SIM_constant_cooling; // ciągłe wychładzanie

    sim_unit SIM_radious_change_proportion; // im większe tym szybciej rosną sfery i na odwrót

    sim_unit SCENE_scale;
    sim_unit SCENE_sphere_separator;
    d3 SCENE_pos_vector;

    sim_unit space_WIDTH;
    sim_unit space_HEIGHT;
    sim_unit space_DEPTH;

    vector<v3<Sim_sphere>> all_spheres_inside_box_ALL_iterations;
    u64 best_memcpy_thread_number;

private:
    u64 check_how_many_spheres_fit_in_this_dimention(sim_unit space_dimention)
    {
        // var(space_dimention);
        // var(SIM_initial_radious);
        // var((2 * SIM_initial_radious));
        // var((space_dimention / (2 * SIM_initial_radious)));
        return (u64)(space_dimention / (2 * SIM_initial_radious));
    }
    tuple<sim_unit, sim_unit> get_smallest_and_largest_temp()
    {
        return {0, 3 * SIM_initial_temperature}; // blocking the accomodation to different ranges of temperatures

        // sim_unit smallest = sim_u(1000000);
        // sim_unit largest = sim_u(0);

        // for (u64 i = 0; i < all_spheres_inside_box.get_total_number(); i++)
        // {
        //     Sim_sphere& sim_sphere = *all_spheres_inside_box.get(i);

        //     sim_unit current_T = sim_sphere.get_T(memory_index.get());

        //     if (current_T < smallest) smallest = current_T;
        //     if (current_T > largest) largest = current_T;
        // }

        // return {smallest, largest};
    }
    // Function to interpolate between two colors based on temperature
    Bmp_RGB get_color_from_temperature(sim_unit temperature, sim_unit smallest_T, sim_unit largest_T, const Bmp_RGB& color_min,
                                       const Bmp_RGB& color_max)
    {
        if (largest_T < temperature) return color_max;
        if (temperature < smallest_T) return color_min;

        // Normalize the temperature value between 0 and 1
        float normalized_temp = (temperature - smallest_T) / (largest_T - smallest_T);

        // var(normalized_temp);

        // Interpolate between the two colors
        int r = static_cast<int>(color_min.get_r() + normalized_temp * (color_max.get_r() - color_min.get_r()));
        int g = static_cast<int>(color_min.get_g() + normalized_temp * (color_max.get_g() - color_min.get_g()));
        int b = static_cast<int>(color_min.get_b() + normalized_temp * (color_max.get_b() - color_min.get_b()));

        return Bmp_RGB(r, g, b);
    }
    sim_unit my_clamp(const sim_unit value, const sim_unit low, const sim_unit high)
    {
        if (high < value) return high;
        if (value < low) return low;
        return value;
    }

public:
    Computation_Box() : best_memcpy_thread_number(0)
    {
        SIM_scale = sim_u(1);
        SIM_initial_sphere_separation = sim_u(1);
        SIM_initial_radious = sim_u(3);
        SIM_initial_temperature = sim_u(100);

        SIM_density = sim_u(1);
        SIM_heat_capacity = sim_u(1);
        SIM_wall_heat_reach = sim_u(50);
        SIM_wall_heat_value = sim_u(20);
        SIM_constant_cooling = sim_u(0);

        SIM_radious_change_proportion = sim_u(0.003);

        SCENE_scale = sim_u(1);
        SCENE_sphere_separator = sim_u(1);
    }
    void fill_space_with_spheres(const sim_unit _space_WIDTH, const sim_unit _space_HEIGHT, const sim_unit _space_DEPTH,
                                 const u64 number_of_iterations)
    {
        SIM_initial_radious *= SIM_scale;

        space_WIDTH = _space_WIDTH * SIM_scale;
        space_HEIGHT = _space_HEIGHT * SIM_scale;
        space_DEPTH = _space_DEPTH * SIM_scale;

        const sim_unit starting_x000 = 0; // SIM_initial_radious
        const sim_unit starting_y000 = 0; // SIM_initial_radious
        const sim_unit starting_z000 = 0; // SIM_initial_radious

        const sim_unit x_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;
        const sim_unit y_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;
        const sim_unit z_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;

        // po koleji wyznaczamy jeden wymiar na raz, a potem iterujemy po tych odnalezionych wymiarach
        u64 how_many_spheres_fit_in_X = check_how_many_spheres_fit_in_this_dimention(space_WIDTH);
        u64 how_many_spheres_fit_in_Y = check_how_many_spheres_fit_in_this_dimention(space_HEIGHT);
        u64 how_many_spheres_fit_in_Z = check_how_many_spheres_fit_in_this_dimention(space_DEPTH);

        nline;
        var(how_many_spheres_fit_in_X);
        var(how_many_spheres_fit_in_Y);
        var(how_many_spheres_fit_in_Z);
        u64 total = how_many_spheres_fit_in_X * how_many_spheres_fit_in_Y * how_many_spheres_fit_in_Z;
        var(total);
        nline;
        var(space_WIDTH);
        var(space_HEIGHT);
        var(space_DEPTH);

        all_spheres_inside_box_ALL_iterations.resize(number_of_iterations);
        for (auto& sim_iterations : all_spheres_inside_box_ALL_iterations)
        {
            sim_iterations.set_sizes(how_many_spheres_fit_in_X, how_many_spheres_fit_in_Y, how_many_spheres_fit_in_Z);
        }

        size_t size_for_one_memcpy = all_spheres_inside_box_ALL_iterations[0].get_total_number() * sizeof(Sim_sphere);

        time_stamp("Checking how many threads perform memcpy best");
        best_memcpy_thread_number =
            // Multithreaded_Memcpy::check_how_many_threads_get_fastest_time(size_for_one_memcpy)
            8;
        var(best_memcpy_thread_number);
        time_stamp("");

        // all_spheres_inside_box.set_sizes(how_many_spheres_fit_in_X, how_many_spheres_fit_in_Y, how_many_spheres_fit_in_Z);

        sim_unit moving_z = starting_z000;
        for (u64 z = 0; z < how_many_spheres_fit_in_Z; z++)
        {
            sim_unit moving_y = starting_y000;
            for (u64 y = 0; y < how_many_spheres_fit_in_Y; y++)
            {
                sim_unit moving_x = starting_x000;
                for (u64 x = 0; x < how_many_spheres_fit_in_X; x++)
                {
                    // Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);
                    Sim_sphere& sim_sphere = *((all_spheres_inside_box_ALL_iterations[0]).get(x, y, z));

                    // clang-format off
                    sim_sphere.init(d3(moving_x, moving_y, moving_z),
                                    SIM_initial_radious
                                    // Randoms::Random_floating_point<double>::random_floating_in_range(SIM_initial_radious, 1.2 * SIM_initial_radious)

                                    ,
                                    SIM_initial_temperature / 2
                                    // Randoms::Random_floating_point<double>::random_floating_in_range(0, SIM_initial_temperature)
                                );

                    // clang-format on
                    moving_x += x_adding;
                }
                moving_y += y_adding;
            }
            moving_z += z_adding;
        }
    }
    tuple<d3, sim_unit> get_distance_and_vec_A_to_B(const d3& posA, const d3& posB)
    {
        d3 vec_from_A_to_B = posB - posA;
        sim_unit distance = d3::distance(vec_from_A_to_B);

        return {vec_from_A_to_B, distance};
    }
    sim_unit sphere_volume(const sim_unit& r) { return (4.0 / 3.0) * M_PI * pow3(r); }
    sim_unit one_over_mass_and_heat_capasity(const sim_unit& r)
    {
        return 1; // simplification

        sim_unit mass = sphere_volume(r) * SIM_density;

        var(sphere_volume(r));
        var(SIM_density);
        var(mass);

        var(SIM_heat_capacity);
        var(mass * SIM_heat_capacity);
        var((1 / (mass * SIM_heat_capacity)));

        FATAL_ERROR("test");

        return (1 / (mass * SIM_heat_capacity));
    }
    bool value_in_between(const sim_unit value, const sim_unit smaller_bound, const sim_unit higher_bound)
    {
        return ((smaller_bound <= value) && (value <= higher_bound));
    }

    sim_unit wall_influence_with_chosen_dimention(const sim_unit dimention_value, const sim_unit max_value)
    {
        // zakładam, że nie ma sytuacji w której przedziały się na siebie na chodzą (zawsze dostajemy ciepło tylko z jednej ze ścian)
        sim_unit reach_start = (max_value - SIM_wall_heat_reach);

        if (value_in_between(dimention_value, 0, SIM_wall_heat_reach))
        {
            sim_unit proportion = sim_u(1) - (dimention_value / SIM_wall_heat_reach);

            return (proportion * SIM_wall_heat_value);
        }
        else if (value_in_between(dimention_value, reach_start, max_value))
        {
            sim_unit proportion = ((dimention_value - reach_start) / SIM_wall_heat_reach);

            return (proportion * SIM_wall_heat_value) * 1.6;
        }
        return 0;
    }

    sim_unit enviroument_influence(const d3& pos)
    {
        sim_unit change_in_T{};

        // change_in_T += wall_influence_with_chosen_dimention(pos.x, space_WIDTH);
        change_in_T += wall_influence_with_chosen_dimention(pos.y, space_HEIGHT);
        // change_in_T += wall_influence_with_chosen_dimention(pos.z, space_DEPTH);

        change_in_T -= SIM_constant_cooling; // womackow - to też można uzależnić od np. odległości do najbliższej ściany
                                             //          - wtedy przekrój będzie lepiej wyglądał, że w środku trzyma ciepło

        return change_in_T;
    }

    // clang-format off
    void transform_to_My_Ray_Tracing_scene(Scene& scene, const u64 iter_index)
    {
        v3<Sim_sphere>& single_iteration_state = all_spheres_inside_box_ALL_iterations[iter_index];

        time_stamp("transform_to_My_Ray_Tracing_scene");
        SCENE_pos_vector =
            d3(sim_u(G::WIDTH / 2 - SIM_initial_radious - 100), sim_u(G::HEIGHT / 2 - SIM_initial_radious - 50), sim_u(-6000 - SIM_initial_radious));

        scene.assign_name("iteration");

        {
            Bmp_RGB light_color = Bmp_RGB(255, 255, 255); // im więcej na minusie, tym bliżej kamery

            scene.add_light(d3(sim_u(G::WIDTH / 2), sim_u(G::HEIGHT / 2 - 200), sim_u(-7000)), (*(const RGB*)(&light_color)));
        }

        auto [smallest_T, largest_T] = get_smallest_and_largest_temp();

        for (u64 i = 0; i < single_iteration_state.get_total_number(); i++)
        {
            Sim_sphere& sim_sphere = *(single_iteration_state.get(i));

            #if defined(CPU) && defined(D_BUFF_SAME_OBJ)
                // womackow -> BARDZO UŻYTECZNE !!! - do i zerknięcia jak jest w środku
                //              + do środka można też zaglądać nie generaując niektórych sfer

                u8 memory_index = (iter_index + 1) % 2;
                d3 scene_pos = sim_sphere.get_position(memory_index)
                    // * SCENE_sphere_separator
                    ; // to chyba rozszerza wszystko - oddzielać sfery od siebie
                sim_unit sphere_r = sim_sphere.get_r(memory_index);
                sim_unit sphere_T = sim_sphere.get_T(memory_index);

            #else

                d3 scene_pos = sim_sphere.get_position();
                sim_unit sphere_r = sim_sphere.get_r();
                sim_unit sphere_T = sim_sphere.get_T();
            #endif

            check_nan(sphere_r);
            check_nan(sphere_T);
            check_nan(scene_pos.x);
            check_nan(scene_pos.y);
            check_nan(scene_pos.z);

            scene_pos.x *= SCENE_scale;
            scene_pos.y *= SCENE_scale;
            scene_pos.z *= SCENE_scale;

            scene_pos.rotate_left_right(d3(0, 0, 0), sim_u(0.5));

            scene_pos.rotate_up_down(d3(0, 0, 0), sim_u(0.5));

            scene_pos.x += SCENE_pos_vector.x;
            scene_pos.y += SCENE_pos_vector.y;
            scene_pos.z += SCENE_pos_vector.z;

            sim_unit scene_r = sphere_r * SCENE_scale;

            Bmp_RGB scene_color =
                get_color_from_temperature(sphere_T, smallest_T, largest_T, Bmp_RGB(0, 0, 255), Bmp_RGB(255, 0, 0))
                // Bmp_RGB(255, 255, 255)
            ;

            // varr(scene_pos.x);
            // varr(scene_pos.y);
            // varr(scene_pos.z);
            // varr(sphere_r);
            // var(sphere_T);

            scene.add_sphere(scene_pos, scene_r, 0.0f, 0.0f, Surface_type::diffuse, (*(const RGB*)(&scene_color)));
        }

        u64 light_limit = (u64)scene.get_lights().size();
        u64 sphere_limit = (u64)scene.get_spheres().size();
        u64 bounce_limit = 5;

        scene.add_scene_detail(light_limit, sphere_limit, bounce_limit);
        scene.add_thread_group(16, 0, 0);
    }

    #if defined(CPU) && defined(D_BUFF_SAME_OBJ)
    void cpu(const u64& number_of_iterations)
    {
        size_t byte_size = all_spheres_inside_box_ALL_iterations[0].get_total_number() * sizeof(Sim_sphere);
        omp_set_nested(true);

        #pragma omp parallel
        {
            Memory_index_cyclic memory_index_cyclic(2);

            for(u64 iter=1; iter < number_of_iterations; iter++)
            {
                #pragma omp for schedule(static)
                for (u64 sphere_index = 0; sphere_index < all_spheres_inside_box_ALL_iterations[0].get_total_number(); sphere_index++)
                {
                    per_sphere(memory_index_cyclic, sphere_index);
                }

                memory_index_cyclic.switch_to_next();
                #pragma omp barrier
                #pragma omp single
                {
                    Multithreaded_Memcpy::cpy
                    (
                        (char*) all_spheres_inside_box_ALL_iterations[iter].data()
                        , (char*) all_spheres_inside_box_ALL_iterations[0].data()
                        
                        , byte_size
                        , best_memcpy_thread_number
                    );
                }
                // setting current iteration values to 0 for correct summing
                #pragma omp for schedule(static)
                for (u64 sphere_index = 0; sphere_index < all_spheres_inside_box_ALL_iterations[0].get_total_number(); sphere_index++)
                {
                    Sim_sphere& current_sphere = *(all_spheres_inside_box_ALL_iterations[0].get(sphere_index));

                    auto& change_current_pos_NEXT_VALUE = current_sphere.get_position(memory_index_cyclic.get_next()); // getting from the same memory
                    auto& change_current_r_NEXT_VALUE = current_sphere.get_r(memory_index_cyclic.get_next());
                    auto& change_current_T_NEXT_VALUE = current_sphere.get_T(memory_index_cyclic.get_next());

                    change_current_pos_NEXT_VALUE = d3(0, 0, 0);
                    change_current_r_NEXT_VALUE = 0;
                    change_current_T_NEXT_VALUE = 0;
                }
                #pragma omp barrier
            }
        }
    }
    #endif

    #if defined(CPU) && defined(D_BUFF_DIFF_OBJ)
    void cpu(const u64& number_of_iterations)
    {
        size_t byte_size = all_spheres_inside_box_ALL_iterations[0].get_total_number() * sizeof(Sim_sphere);
        omp_set_nested(true);

        #pragma omp parallel
        {
            Memory_index_cyclic memory_index_cyclic(2);

            for(u64 iter=2; iter < number_of_iterations; iter++)
            {
                #pragma omp for schedule(static)
                for (u64 sphere_index = 0; sphere_index < all_spheres_inside_box_ALL_iterations[0].get_total_number(); sphere_index++)
                {
                    per_sphere(memory_index_cyclic, sphere_index);
                }

                memory_index_cyclic.switch_to_next();
                #pragma omp barrier
                #pragma omp single
                {
                    Multithreaded_Memcpy::cpy
                    (
                        (char*) all_spheres_inside_box_ALL_iterations[iter].data()
                        , (char*) all_spheres_inside_box_ALL_iterations[memory_index_cyclic.get_next()].data()
                        
                        , byte_size
                        , best_memcpy_thread_number
                    );

                    // setting current iteration values to 0 for correct summing
                    memset
                    (
                        all_spheres_inside_box_ALL_iterations[memory_index_cyclic.get_next()].data()
                        , 0
                        , byte_size
                    );
                }
                #pragma omp barrier
            }
        }
    }
    #endif

    #if defined(CPU) && defined(N_BUFF)
    void cpu(const u64& number_of_iterations)
    {
        #pragma omp parallel
        {
            Memory_index_cyclic memory_index_cyclic(number_of_iterations);

            for(u64 iter=1; iter < number_of_iterations; iter++)
            {
                #pragma omp for schedule(static)
                for (u64 sphere_index = 0; sphere_index < all_spheres_inside_box_ALL_iterations[0].get_total_number(); sphere_index++)
                {
                    per_sphere(memory_index_cyclic, sphere_index);
                }

                memory_index_cyclic.switch_to_next();
                #pragma omp barrier
            }
        }
    }
    #endif

    #if defined(GPU)
    void gpu()
    {
        // MAIN //

        int BLOCK_SIZE = 64; // 32 - 64 - 128 - 256
        int NUMBER_OF_BLOCKS = PIXEL_ARRAY_SIZE / BLOCK_SIZE + 1; // zeby pokrywalo wszystko -> przez to musi byc warunek
        size_t byte_size;

        byte_size = sizeof(*this);
        Computation_Box* dev_Computation_Box{};
        CCE(cudaMalloc((void**)&dev_Computation_Box, byte_size));
        CCE(cudaMemcpy(dev_Computation_Box, this, byte_size, cudaMemcpyHostToDevice));

        byte_size = all_spheres_inside_box_ALL_iterations[0].get_total_number() * sizeof(Sim_sphere);
        RGB* dev_My_Pixels{};                                                               // v3<Sim_sphere> -> używajmy jak raw tablicy
        CCE(cudaMalloc((void**)&dev_My_Pixels, size));                                      // przekopiujemy każdy blok pamięci i będzie jego dev_ptr
        CCE(cudaMemcpy(dev_My_Pixels, get_my_pixel(), size, cudaMemcpyHostToDevice));       // którego wpiszemy do innej tablicy
                                                                                            // w ten sposób na GPU, będziemy mieli wiele instancji
        Light_point* dev_Current_Scene_lights{};
        {
            Light_point* host_ptr = G::Render::current_scene->get_lights_ptr();
            auto& host_vec_ref = G::Render::current_scene->get_lights();
            size = sizeof(Light_point) * host_vec_ref.size();
            CCE(cudaMalloc((void**)&dev_Current_Scene_lights, size));
            CCE(cudaMemcpy(dev_Current_Scene_lights, host_ptr, size, cudaMemcpyHostToDevice));
        }

        Sphere* dev_Current_Scene_spheres{};
        {
            Sphere* host_ptr = G::Render::current_scene->get_spheres_ptr();
            auto& host_vec_ref = G::Render::current_scene->get_spheres();
            size = sizeof(Sphere) * host_vec_ref.size();
            CCE(cudaMalloc((void**)&dev_Current_Scene_spheres, size));
            CCE(cudaMemcpy(dev_Current_Scene_spheres, host_ptr, size, cudaMemcpyHostToDevice));
        }

        details* dev_Current_Scene_details{};
        {
            details* host_ptr = G::Render::current_scene->get_details_ptr();
            auto& host_vec_ref = G::Render::current_scene->get_details();
            size = sizeof(details) * host_vec_ref.size();
            CCE(cudaMalloc((void**)&dev_Current_Scene_details, size));
            CCE(cudaMemcpy(dev_Current_Scene_details, host_ptr, size, cudaMemcpyHostToDevice));
        }

        CCE(cudaDeviceSynchronize());
        for (int i = 0; i < G::REP_NUMBER; i++)
        {
            Timer kernel_timer;

            kernel_timer.start();
            {
                render_kernel<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(
                    (G::WIDTH / 2), (G::HEIGHT / 2), (-10000.0), WIDTH, PIXEL_ARRAY_SIZE, dev_Renderer, dev_My_Pixels,
                    dev_Current_Scene, dev_Current_Scene_lights, dev_Current_Scene_spheres, dev_Current_Scene_details);
                CCE(cudaDeviceSynchronize());
            }
            kernel_timer.end();

            G::Render::current_scene_stats->get_stats().push_whole(kernel_timer.get_all_in_nano());
        }

        CCE(cudaMemcpy(get_my_pixel(), dev_My_Pixels, sizeof(RGB) * PIXEL_ARRAY_SIZE, cudaMemcpyDeviceToHost));

        // zwalnia od razu wszystko
        // CCE(cudaDeviceReset());

        CCE(cudaFree(dev_Renderer));
        CCE(cudaFree(dev_My_Pixels));
        CCE(cudaFree(dev_Current_Scene));
        CCE(cudaFree(dev_Current_Scene_lights));
        CCE(cudaFree(dev_Current_Scene_spheres));
        CCE(cudaFree(dev_Current_Scene_details));
    }
    #endif

    GPU_LINE(__device__)
    void per_sphere(const Memory_index_cyclic& memory_index, const u64& sphere_index)
    {
        #if defined(CPU) && defined(D_BUFF_SAME_OBJ)
            Sim_sphere& current_sphere = *(all_spheres_inside_box_ALL_iterations[0].get(sphere_index));             // always using 0 index
            
            const auto& current_pos = current_sphere.get_position(memory_index.get());
            const auto& current_r = current_sphere.get_r(memory_index.get());
            const auto& current_T = current_sphere.get_T(memory_index.get());

            auto& change_current_pos_NEXT_VALUE = current_sphere.get_position(memory_index.get_next()); // getting from the same memory
            auto& change_current_r_NEXT_VALUE = current_sphere.get_r(memory_index.get_next());
            auto& change_current_T_NEXT_VALUE = current_sphere.get_T(memory_index.get_next());

        #else
        
            Sim_sphere& current_sphere = *(all_spheres_inside_box_ALL_iterations[memory_index.get()].get(sphere_index));
            Sim_sphere& current_sphere_NEXT_VALUE = *(all_spheres_inside_box_ALL_iterations[memory_index.get_next()].get(sphere_index));
            
            const auto& current_pos = current_sphere.get_position();
            const auto& current_r = current_sphere.get_r();
            const auto& current_T = current_sphere.get_T();

            auto& change_current_pos_NEXT_VALUE = current_sphere_NEXT_VALUE.get_position(); // getting from different memory
            auto& change_current_r_NEXT_VALUE = current_sphere_NEXT_VALUE.get_r();
            auto& change_current_T_NEXT_VALUE = current_sphere_NEXT_VALUE.get_T();
        
        #endif

        sim_unit running_sum_of_heat_change_due_to_radiation{};
        sim_unit running_sum_of_heat_change_due_to_conductivity{};

        d3 sphere_correction = d3(0, 0, 0);
        for (u64 other_i = 0; other_i < all_spheres_inside_box_ALL_iterations[0].get_total_number(); other_i++)
        {
            if (sphere_index == other_i) continue;

            #if defined(CPU) && defined(D_BUFF_SAME_OBJ)
                Sim_sphere& other_sp = *(all_spheres_inside_box_ALL_iterations[0].get(other_i));
                
                const auto& other_pos = other_sp.get_position(memory_index.get());
                const auto& other_r = other_sp.get_r(memory_index.get());
                const auto& other_T = other_sp.get_T(memory_index.get());
            #else

                Sim_sphere& other_sp = *(all_spheres_inside_box_ALL_iterations[memory_index.get()].get(other_i));                

                const auto& other_pos = other_sp.get_position();
                const auto& other_r = other_sp.get_r();
                const auto& other_T = other_sp.get_T();
            #endif
            
            const sim_unit r_sum = current_r + other_r;
            const sim_unit r_smaller = std::min(current_r, other_r);
            const sim_unit r_bigger = std::max(current_r, other_r);

            auto [vec_from_A_to_B, distance] = get_distance_and_vec_A_to_B(current_pos, other_pos);

            #ifdef TEMP_DISTRIBUTION

                // jeśli dystans się zgadza sphere_index jest np. mniejszy niż 2 * r Current_sfery to bierzemy sferę do obliczeń
                // tak samo jak collision, sumujemy wszystkie zmiany albo jakiś inny mechanizm sphere_index pod koniec pętli zmieniamy następny memory index
                // tak samo promień, zmieniamy tego następnego

                // dla GPU - można wykonać najpierw wszystkie kroki iteracyjne sphere_index tylko synchronizować je po przeiterowaniu sphere_index wykonaniu jednej per_sphere
                // + zapis stanu pod koniec do pamięci
                // sam v3 (będzie znana ilość iteracji, więc przed startem się to alokuje sphere_index później zczyta jako tablica v3)

                // WSZYSTKIE WZORY //
                // https://chatgpt.com/c/67a8e634-cec8-8009-8a2f-4942550ab331

                if (distance < (r_bigger * sim_u(2)))
                {
                    // - ogarniamy ile energii trafia w sferę w zależności od jej odległości sphere_index rozmiaru -

                    // AKTUALNIE - ten sposób przekazuje także kiedy sfery się nachodzą //

                    // OTHER - emmits //
                    // CURRENT - calculates how much it gets //

                    const sim_unit tan_value = current_r / distance;
                    const sim_unit alfa = 2 * atan(tan_value);
                    const sim_unit percent_of_captured_energy = alfa / (2 * M_PI);

                    sim_unit other_sphere_total_emmited_energy = 10;

                    running_sum_of_heat_change_due_to_radiation += other_sphere_total_emmited_energy * percent_of_captured_energy;
                }

                if (distance < r_sum)
                {
                    // - ogarniamy powierzchnię tego styku dla różnych rozmiarów sfer -

                    // running_sum_of_heat_change_due_to_conductivity +=
                }

            #endif // TEMP_DISTRIBUTION

            #ifdef COLLISION_RESOLUTION
                if (distance < r_sum)
                {
                    vec_from_A_to_B.normalize();
                    vec_from_A_to_B.negate();
                    
                    sim_unit correction = r_sum - distance;
                    correction *= (current_r) / r_sum; // womackow - taking size into account when moving

                    vec_from_A_to_B *= correction;
                    sphere_correction += vec_from_A_to_B;
                }
            #endif // COLLISION_RESOLUTION

            #ifdef VOLUME_TRANSFER
                // tutaj będzie przeszukiwanie najgorętszej sfery w okolicy

                // trzeba złapać do niej pointer, bo będziemy musieli jej później nadpisać wartości

            #endif // VOLUME_TRANSFER
        }

        #ifdef COLLISION_RESOLUTION

            d3 new_pos = current_pos + sphere_correction;

            // wall correction
            {
                new_pos.x = my_clamp(new_pos.x, 0, space_WIDTH);
                new_pos.y = my_clamp(new_pos.y, 0, space_HEIGHT);
                new_pos.z = my_clamp(new_pos.z, 0, space_DEPTH);
            }

            // change_current_pos_NEXT_VALUE = current_pos; // no changes

            // TO NEXT ITERATION //
            change_current_pos_NEXT_VALUE = new_pos;

        #endif // COLLISION_RESOLUTION

        #ifdef TEMP_DISTRIBUTION
            
            sim_unit new_T = current_T +
                (
                    one_over_mass_and_heat_capasity(current_r) *
                    (
                        running_sum_of_heat_change_due_to_conductivity +
                        running_sum_of_heat_change_due_to_radiation

                        + enviroument_influence(current_pos)
                    )
                );
            
            new_T = std::max(sim_u(0), new_T); // keeping temp above absolute zero
            
            // change_current_T_NEXT_VALUE = current_T; // no changes

            // TO NEXT ITERATION //
            change_current_T_NEXT_VALUE = new_T;

        #endif // TEMP_DISTRIBUTION

        #ifdef VOLUME_TRANSFER

            // ZARÓWNO zmiana CURRENT_next_value musi być zrobiona atomowa
            // tak samo OTHER_next_value musi

            sim_unit new_r = current_r *
                (
                    1 + (
                            SIM_radious_change_proportion *
                            (new_T - current_T)
                        )
                ); // womackow - decyzja o tym czy usuwamy sferę czy nie, może być tylko pole w SIM_sphere
                                            // sphere_index jeśli na nie trafimy to skip
                                            // to nie działa dla różnych punktów startowych, jeśli sfery są zainicjowane różnymi temperaturami,
                                            // potrzebujemy jakiejś funcji, co jasno określa jakich rozmiarów w danej temperaturz ma być sfera

            new_r = std::min(sim_u(1.5 * SIM_initial_radious), new_r); // womackow - tylko na teraz żeby nie rozbuchały,
                                                                // później to trzeba będzie zrobić tak, że nawet w tych ciepłych
                                                                // jedne rosną, a inne maleją, bo są wchłaniane przez sąsiadów
                                                                // to bez sensu jeśli to prostu znikąd rosną, ta masa musi się skądś wziąć
            
            
            
            // change_current_r_NEXT_VALUE = current_r; // no changes

            // TO NEXT ITERATION //

            // CPU - #pragma omp atomic
            // GPU - atomidAdd(&adding_to, value)

            change_current_r_NEXT_VALUE += new_r; // NEXT_VALUE has to be zero - adding because r must allow other spheres to modify it's value
                                           // current will add his
                                           // while others might add their portions
            
        #endif // VOLUME_TRANSFER
    }
};