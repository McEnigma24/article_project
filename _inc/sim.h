#include "Nano_Timer.h"
#include "visualizer.h"

class Sim_sphere
{
    d3 position;
    unit r;
    unit T;

public:
    Sim_sphere() { memset(this, 0, sizeof(*this)); }

    d3 get_position() const { return position; }
    unit get_r() const { return r; }
    unit get_T() const { return T; }

    void init(const d3& _position, const unit& _r, const unit& _T)
    {
        position = _position;
        r = _r;
        T = _T;
    }
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

    T get(u64 x, u64 y, u64 z) const
    {
        BOUND_CHECKS_V3(if (!bound_checks(x, y, z)) return T();)

        return buffer[d3_to_d1(x, y, z)];
    }

    T* get(u64 x, u64 y, u64 z)
    {
        BOUND_CHECKS_V3(if (!bound_checks(x, y, z)) return nullptr;)

        return &buffer[d3_to_d1(x, y, z)];
    }
};

class Computation_Box
{
    unit SIM_scale;
    unit initial_radious;
    unit initial_temperature;

    unit SCENE_scale;
    d3 SCENE_pos_vector;
    // rotation

    v3<Sim_sphere> all_spheres_inside_box;

private:
    u64 check_how_many_spheres_fit_in_this_dimention(unit space_dimention) { return (u64)(space_dimention / (2 * initial_radious)); }

    tuple<unit, unit> get_smallest_and_largest_temp()
    {
        unit smallest = u(1000000);
        unit largest = u(0);

        for (u64 z{}; z < all_spheres_inside_box.get_depth(); z++)
            for (u64 y{}; y < all_spheres_inside_box.get_height(); y++)
                for (u64 x{}; x < all_spheres_inside_box.get_width(); x++)
                {
                    Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);

                    unit current_T = sim_sphere.get_T();

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

        // Interpolate between the two colors
        int r = static_cast<int>(color_min.get_r() + normalized_temp * (color_max.get_r() - color_min.get_r()));
        int g = static_cast<int>(color_min.get_g() + normalized_temp * (color_max.get_g() - color_min.get_g()));
        int b = static_cast<int>(color_min.get_b() + normalized_temp * (color_max.get_b() - color_min.get_b()));

        return Bmp_RGB(r, g, b);
    }

public:
    Computation_Box()
    {
        SIM_scale = u(1);
        initial_radious = u(22);
        initial_temperature = u(273.15);

        unit SCENE_scale = u(1);
        var(G::WIDTH);
        var(G::HEIGHT);

        var(u(G::WIDTH / 2 - 250 + 50));
        var(u(G::HEIGHT / 2 - 50));
        var(u(-5000));
        d3 SCENE_pos_vector = d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-5000));
    }

    void fill_space_with_spheres(unit _space_WIDTH, unit _space_HEIGHT, unit _space_DEPTH)
    {
        initial_radious *= SIM_scale;

        unit space_WIDTH = _space_WIDTH * SIM_scale;
        unit space_HEIGHT = _space_HEIGHT * SIM_scale;
        unit space_DEPTH = _space_DEPTH * SIM_scale;

        const unit starting_x000 = initial_radious;
        const unit starting_y000 = initial_radious;
        const unit starting_z000 = initial_radious;

        const unit x_adding = 2 * initial_radious;
        const unit y_adding = 2 * initial_radious;
        const unit z_adding = 2 * initial_radious;

        // po koleji wyznaczamy jeden wymiar na raz, a potem iterujemy po tych odnalezionych wymiarach
        u64 how_many_spheres_fit_in_X = check_how_many_spheres_fit_in_this_dimention(space_WIDTH);
        u64 how_many_spheres_fit_in_Y = check_how_many_spheres_fit_in_this_dimention(space_HEIGHT);
        u64 how_many_spheres_fit_in_Z = check_how_many_spheres_fit_in_this_dimention(space_DEPTH);

        nline;
        var(how_many_spheres_fit_in_X);
        var(how_many_spheres_fit_in_Y);
        var(how_many_spheres_fit_in_Z);

        all_spheres_inside_box.set_sizes(how_many_spheres_fit_in_X, how_many_spheres_fit_in_Y, how_many_spheres_fit_in_Z);

        unit moving_z = starting_z000;
        for (u64 z{}; z < how_many_spheres_fit_in_Z; z++)
        {
            unit moving_y = starting_y000;
            for (u64 y{}; y < how_many_spheres_fit_in_Y; y++)
            {
                unit moving_x = starting_x000;
                for (u64 x{}; x < how_many_spheres_fit_in_X; x++)
                {
                    all_spheres_inside_box.get(x, y, z)->init(d3(moving_x, moving_y, moving_z), initial_radious, initial_temperature);

                    moving_x += x_adding;
                }
                moving_y += y_adding;
            }
            moving_z += z_adding;
        }
    }

    void temp_dist() {}

    void collision_resolution() {}

    void transform_to_My_Ray_Tracing_scene(Scene& scene)
    {
        scene.assign_name("iteration");

        {
            Bmp_RGB light_color = Bmp_RGB(255, 255, 255);
            scene.add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-5000)), (*(const RGB*)(&light_color)));
        }

        auto [smallest_T, largest_T] = get_smallest_and_largest_temp();

        for (u64 z{}; z < all_spheres_inside_box.get_depth(); z++)
            for (u64 y{}; y < all_spheres_inside_box.get_height(); y++)
                for (u64 x{}; x < all_spheres_inside_box.get_width(); x++)
                {
                    Sim_sphere& sim_sphere = *all_spheres_inside_box.get(x, y, z);

                    d3 scene_pos = sim_sphere.get_position() * SCENE_scale;

                    var(scene_pos.x);
                    var(scene_pos.y);
                    var(scene_pos.z);

                    var(SCENE_pos_vector.x);
                    var(SCENE_pos_vector.y);
                    var(SCENE_pos_vector.z);

                    scene_pos.x += SCENE_pos_vector.x;
                    scene_pos.y += SCENE_pos_vector.y;
                    scene_pos.z += SCENE_pos_vector.z;

                    var(scene_pos.x);
                    var(scene_pos.y);
                    var(scene_pos.z);

                    unit scene_r = sim_sphere.get_r() * SCENE_scale;

                    Bmp_RGB scene_color =
                        get_color_from_temperature(sim_sphere.get_T(), smallest_T, largest_T, Bmp_RGB(0, 0, 255), Bmp_RGB(255, 0, 0));

                    scene.add_sphere(scene_pos, scene_r, 0.0f, 0.0f, Surface_type::diffuse,
                                     // Bmp_RGB(0, 255, 0)
                                     (*(const RGB*)(&scene_color)));
                }

        u64 light_limit = (u64)scene.get_lights().size();
        u64 sphere_limit = (u64)scene.get_spheres().size();
        u64 bounce_limit = 5;

        scene.add_scene_detail(light_limit, sphere_limit, bounce_limit);
        scene.add_thread_group(16, 0, 0);
    }
};