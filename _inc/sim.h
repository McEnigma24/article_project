#include "Nano_Timer.h"
#include "visualizer.h"
#include <omp.h>

class Sim_sphere
{
    d3 position[2];
    unit r[2];
    unit T[2];

public:
    Sim_sphere() { memset(this, 0, sizeof(*this)); }

    d3 get_position(u8 index) const { return position[index]; }
    unit get_r(u8 index) const { return r[index]; }
    unit get_T(u8 index) const { return T[index]; }

    void init(const d3& _position, const unit& _r, const unit& _T)
    {
        position[0] = _position;
        position[1] = _position;
        r[0] = _r;
        r[1] = _r;
        T[0] = _T;
        T[1] = _T;
    }

    void set_new_position(const d3& new_pos, u8 index) { position[index] = new_pos; }
    void set_new_T(const unit& _T, u8 index) { T[index] = _T; }
};

// na początku można zrobić z tymi prymitywnymi RÓWNOLEGŁYMI //
// temp_dist                -> dla wszystkich par (nowe pary, jeśli ilość sfer się zmieni)
//                              ograniczone o jakiś dystans
// collision_resolution      -> wszystkie, i poprawki tylko dla tych co się nachodzą

#define BOUND_CHECKS_V3(...) __VA_ARGS__
// #define BOUND_CHECKS_V3(...)

template <typename T>
class v3
{
    u64 WIDTH;
    u64 HEIGHT;
    u64 DEPTH;
    vector<T> buffer;

    bool bound_checks(u64 x, u64 y, u64 z) { return (0 <= x && x < WIDTH) && (0 <= y && y < HEIGHT) && (0 <= z && z < DEPTH); }
    u64 d3_to_d1(u64 x, u64 y, u64 z) const { return x + y * (HEIGHT) + z * (WIDTH * HEIGHT); }

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

class Memory_index
{
    u8 index;

public:
    Memory_index() : index(0) {}

    u8 get() const { return index; }
    u8 get_next() const { return (index + 1) % 2; }

    void switch_to_next() { index = (index + 1) % 2; }
};

class Computation_Box
{
    unit SIM_scale;
    unit SIM_initial_sphere_separation;
    unit SIM_initial_radious;
    unit SIM_initial_temperature;

    unit SCENE_scale;
    unit SCENE_sphere_separator;
    d3 SCENE_pos_vector;

    unit space_WIDTH;
    unit space_HEIGHT;
    unit space_DEPTH;

    v3<Sim_sphere> all_spheres_inside_box;

private:
    u64 check_how_many_spheres_fit_in_this_dimention(unit space_dimention) { return (u64)(space_dimention / (2 * SIM_initial_radious)); }

    tuple<unit, unit> get_smallest_and_largest_temp(const Memory_index& memory_index)
    {
        unit smallest = u(1000000);
        unit largest = u(0);

        for (u64 z = 0; z < all_spheres_inside_box.get_depth(); z++)
            for (u64 y = 0; y < all_spheres_inside_box.get_height(); y++)
                for (u64 x = 0; x < all_spheres_inside_box.get_width(); x++)
                {
                    Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);

                    unit current_T = sim_sphere.get_T(memory_index.get());

                    if (current_T < smallest) smallest = current_T;
                    if (current_T > largest) largest = current_T;
                }

        return {smallest, largest};
    }

    // Function to interpolate between two colors based on temperature
    Bmp_RGB get_color_from_temperature(unit temperature, unit smallest_T, unit largest_T, const Bmp_RGB& color_min, const Bmp_RGB& color_max)
    {
        // Normalize the temperature value between 0 and 1
        float normalized_temp = (temperature - smallest_T) / (largest_T - smallest_T);

        // var(normalized_temp);

        // Interpolate between the two colors
        int r = static_cast<int>(color_min.get_r() + normalized_temp * (color_max.get_r() - color_min.get_r()));
        int g = static_cast<int>(color_min.get_g() + normalized_temp * (color_max.get_g() - color_min.get_g()));
        int b = static_cast<int>(color_min.get_b() + normalized_temp * (color_max.get_b() - color_min.get_b()));

        return Bmp_RGB(r, g, b);
    }

    unit my_clamp(unit value, unit low, unit high)
    {
        if (high < value) return high;
        if (value < low) return low;
        return value;
    }

public:
    Computation_Box()
    {
        SIM_scale = u(1);
        SIM_initial_sphere_separation = u(0.8);
        SIM_initial_radious = u(3);
        SIM_initial_temperature = u(273.15);

        SCENE_scale = u(1);
        SCENE_sphere_separator = u(1);
    }

    void fill_space_with_spheres(unit _space_WIDTH, unit _space_HEIGHT, unit _space_DEPTH)
    {
        SIM_initial_radious *= SIM_scale;

        space_WIDTH = _space_WIDTH * SIM_scale;
        space_HEIGHT = _space_HEIGHT * SIM_scale;
        space_DEPTH = _space_DEPTH * SIM_scale;

        const unit starting_x000 = SIM_initial_radious;
        const unit starting_y000 = SIM_initial_radious;
        const unit starting_z000 = SIM_initial_radious;

        const unit x_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;
        const unit y_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;
        const unit z_adding = 2 * SIM_initial_radious * SIM_initial_sphere_separation;

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

        all_spheres_inside_box.set_sizes(how_many_spheres_fit_in_X, how_many_spheres_fit_in_Y, how_many_spheres_fit_in_Z);

        unit moving_z = starting_z000;
        for (u64 z = 0; z < how_many_spheres_fit_in_Z; z++)
        {
            unit moving_y = starting_y000;
            for (u64 y = 0; y < how_many_spheres_fit_in_Y; y++)
            {
                unit moving_x = starting_x000;
                for (u64 x = 0; x < how_many_spheres_fit_in_X; x++)
                {
                    Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);

                    sim_sphere.init(d3(moving_x, moving_y, moving_z),
                                    SIM_initial_radious
                                    // ,SIM_initial_temperature
                                    , );

                    moving_x += x_adding;
                }
                moving_y += y_adding;
            }
            moving_z += z_adding;
        }
    }

    tuple<d3, unit> get_distance_and_vec_A_to_B(const d3& posA, const d3& posB)
    {
        d3 vec_from_A_to_B = posB - posA;
        unit distance = d3::distance(vec_from_A_to_B);

        return {vec_from_A_to_B, distance};
    }

    void per_sphere(const Memory_index& memory_index, const u64& i)
    {
        Sim_sphere& current_sphere = *all_spheres_inside_box.get(i);
        d3 sphere_correction = d3(0, 0, 0);

        for (u64 other_i = 0; other_i < all_spheres_inside_box.get_total_number(); other_i++)
        {
            if (i == other_i) continue;
            Sim_sphere& other_sp = *all_spheres_inside_box.get(other_i);

            auto [vec_from_A_to_B, distance] =
                get_distance_and_vec_A_to_B(current_sphere.get_position(memory_index.get()), other_sp.get_position(memory_index.get()));

#ifdef TEMP_DISTRIBUTION

            // jeśli dystans się zgadza i jest np. mniejszy niż 2 * r Current_sfery to bierzemy sferę do obliczeń
            // tak samo jak collision, sumujemy wszystkie zmiany albo jakiś inny mechanizm i pod koniec pętli zmieniamy następny memory index
            // tak samo promień, zmieniamy tego następnego

            // dla GPU - można wykonać najpierw wszystkie kroki iteracyjne i tylko synchronizować je po przeiterowaniu i wykonaniu jednej per_sphere
            // + zapis stanu pod koniec do pamięci
            // sam v3 (będzie znana ilość iteracji, więc przed startem się to alokuje i później zczyta jako tablica v3)

#endif

#ifdef COLLISION_RESOLUTION
            if (distance < (current_sphere.get_r(memory_index.get()) + other_sp.get_r(memory_index.get())))
            {
                vec_from_A_to_B.normalize();
                vec_from_A_to_B.negate();

                unit correction = (current_sphere.get_r(memory_index.get()) + other_sp.get_r(memory_index.get())) - distance;
                // correction *= ; // później uwzględniamy proporcję przesunięcia do rozmiaru sfer -> rA + rB

                vec_from_A_to_B *= correction;

                sphere_correction += vec_from_A_to_B;
            }
#endif // COLLISION_RESOLUTION
        }

#ifdef COLLISION_RESOLUTION

        d3 old_pos = current_sphere.get_position(memory_index.get());
        d3 new_pos = old_pos + sphere_correction;

        // sprawdzanie ze ścianami
        {
            new_pos.x = my_clamp(new_pos.x, 0, space_WIDTH);
            new_pos.y = my_clamp(new_pos.y, 0, space_HEIGHT);
            new_pos.z = my_clamp(new_pos.z, 0, space_DEPTH);
        }

        // nadpisujemy następną pozycję
        current_sphere.set_new_position(new_pos, memory_index.get_next());
#endif // COLLISION_RESOLUTION
    }

    void iteration_step(const Memory_index& memory_index)
    {
#ifdef CPU
# pragma omp parallel for schedule(static)
        for (u64 i = 0; i < all_spheres_inside_box.get_total_number(); i++)
        {
            per_sphere(memory_index, i);
        }
#endif // CPU
    }

    void transform_to_My_Ray_Tracing_scene(Scene& scene, const Memory_index& memory_index)
    {
        time_stamp("transform_to_My_Ray_Tracing_scene");
        SCENE_pos_vector =
            d3(u(G::WIDTH / 2 - SIM_initial_radious - 50), u(G::HEIGHT / 2 - SIM_initial_radious - 30), u(-6000 - SIM_initial_radious));

        scene.assign_name("iteration");

        {
            Bmp_RGB light_color = Bmp_RGB(255, 255, 255); // im więcej na minusie, tym bliżej kamery

            scene.add_light(d3(u(G::WIDTH / 2), u(G::HEIGHT / 2 - 200), u(-7000)), (*(const RGB*)(&light_color)));
        }

        auto [smallest_T, largest_T] = get_smallest_and_largest_temp(memory_index);

        for (u64 z = 0; z < all_spheres_inside_box.get_depth(); z++)
            for (u64 y = 0; y < all_spheres_inside_box.get_height(); y++)
                for (u64 x = 0; x < all_spheres_inside_box.get_width(); x++)
                {
                    Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);

                    // womackow -> BARDZO UŻYTECZNE !!! - do i zerknięcia jak jest w środku
                    //              + do środka można też zaglądać nie generaując niektórych sfer

                    d3 scene_pos = sim_sphere.get_position(memory_index.get()) *
                                   SCENE_sphere_separator; // to chyba rozszerza wszystko - oddzielać sfery od siebie

                    scene_pos.rotate_left_right(d3(0, 0, 0), u(0.5));

                    scene_pos.rotate_up_down(d3(0, 0, 0), u(0.5));

                    scene_pos.x += SCENE_pos_vector.x;
                    scene_pos.y += SCENE_pos_vector.y;
                    scene_pos.z += SCENE_pos_vector.z;

                    unit scene_r = sim_sphere.get_r(memory_index.get()) * SCENE_scale;

                    Bmp_RGB scene_color =
                        // get_color_from_temperature(sim_sphere.get_T(memory_index.get()), smallest_T, largest_T, Bmp_RGB(0, 0, 255), Bmp_RGB(255, 0,
                        // 0))
                        Bmp_RGB(255, 255, 255);

                    scene.add_sphere(scene_pos, scene_r, 0.0f, 0.0f, Surface_type::diffuse, (*(const RGB*)(&scene_color)));
                }

        u64 light_limit = (u64)scene.get_lights().size();
        u64 sphere_limit = (u64)scene.get_spheres().size();
        u64 bounce_limit = 5;

        scene.add_scene_detail(light_limit, sphere_limit, bounce_limit);
        scene.add_thread_group(16, 0, 0);
    }
};