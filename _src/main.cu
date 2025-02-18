#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "openMP_test.h"
#include "visualizer.h"

#include "Nano_Timer.h"

class Sim_sphere
{
    d3 position;
    unit r;
    unit T;

public:
    Sim_sphere() { memset(this, 0, sizeof(*this)); }

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
// collision_resolutin      -> wszystkie, i poprawki tylko dla tych co się nachodzą

// mój template - UTILS -> CORE, time stamp też w namespace CORE

#define BOUND_CHECKS_V3(...) __VA_ARGS__
// #define BOUND_CHECKS_V3(...)

template <typename T>
class v3
{
    u64 WIDTH;
    u64 HEIGHT;
    u64 DEPTH;
    vector<T> buffer;

    bool bound_checks(u64 x, u64 y, u64 z)
    {
        return (0 <= x && x < WIDTH) && (0 <= y && y < HEIGHT) && (0 <= z && z < DEPTH);
    }
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

    v3<Sim_sphere> all_spheres_inside_box;

public:
    Computation_Box()
    {
        SIM_scale = u(1);
        initial_radious = u(10);
        initial_temperature = u(273.15);
    }
    u64 check_how_many_spheres_fit_in_this_dimention(unit space_dimention)
    {
        return (u64)(space_dimention / 2 * initial_radious);
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

        all_spheres_inside_box.set_sizes(how_many_spheres_fit_in_X, how_many_spheres_fit_in_Y,
                                         how_many_spheres_fit_in_Z);

        unit moving_z = starting_z000;
        for (u64 z{}; z < how_many_spheres_fit_in_Z; z++)
        {
            unit moving_y = starting_y000;
            for (u64 y{}; y < how_many_spheres_fit_in_Y; y++)
            {
                unit moving_x = starting_x000;
                for (u64 x{}; x < how_many_spheres_fit_in_X; x++)
                {
                    all_spheres_inside_box.get(x, y, z)->init(d3(moving_x, moving_y, moving_z), initial_radious,
                                                              initial_temperature);

                    moving_x += x_adding;
                }
                moving_y += y_adding;
            }
            moving_z += z_adding;
        }
    }

    Scene transform_to_My_Ray_Tracing_scene() {}
};

int main(int argc, char* argv[])
{
    srand(time(NULL));

    Computation_Box box;
    time_stamp("Box created");
    box.fill_space_with_spheres(u(100), u(100), u(100));
    time_stamp("Box filled with spheres");

    const int WIDTH = 1000;
    const int HEIGHT = 1000;

    // const int WIDTH = 1024;
    // const int HEIGHT = 768;

    INPUT_scene_OUTPUT_movie scene_renderer(WIDTH, HEIGHT, "movie.mp4", 5);

    if (false)
    {
        Scene scene;
        // Setuper::setup_scene_0(&scene, "first");

        unit z_plane = u(100);
        scene.assign_name("scene");

        if (true)
        {
            unit z_plane = u(100);

            scene.add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-5000.0)), RGB(255, 255, 255));

            scene.add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-150))
                            //, RGB(255, 0, 0));
                            ,
                            RGB(255, 255, 255));

            scene.add_sphere(d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane), u(100), // DEAD CENTER
                             0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 0));

            scene.add_sphere(d3(u(G::WIDTH / 2 + u(100)), u(G::HEIGHT / 2), z_plane + u(100)), u(100), // DEAD CENTER
                             0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 0));

            scene.add_sphere(d3(u(G::WIDTH / 2 + u(75)), u(G::HEIGHT / 2 - 200),
                                z_plane // little higher - shadow maker
                                    - u(125)
                                //- u(50)
                                ),
                             u(40), 0.0f, 0.0f, Surface_type::diffuse, RGB(175, 255, 0));

            scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300, z_plane + u(-150)),
                             u(100), // bottom - right
                             0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 255));

            scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300 + 50, z_plane), u(150), // bottom - right
                             0.0f, 0.0f, Surface_type::diffuse, RGB(0, 0, 255));

            scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) - 300, z_plane), u(200), // top	  - right
                             0.0f, 0.0f, Surface_type::diffuse, RGB(255, 0, 255));

            scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) + 300, z_plane), u(100), // bottom - left
                             0.0f, 0.0f, Surface_type::diffuse, RGB(255, 255, 0));

            scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane), u(100), // top	  -	left
                             0.0f, 0.0f, Surface_type::diffuse, RGB(255, 255, 255));

            scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane - u(25)),
                             u(80), // top	  -	left
                             0.0f, 0.0f, Surface_type::diffuse, RGB(255, 0, 0));
        }

        // scene.add_light
        // (
        //     d3(u(WIDTH / 2), u(HEIGHT / 2), z_plane)
        //     ,RGB(255, 255, 255)
        // );

        // scene.add_sphere
        // (
        //     d3(u(WIDTH / 2), u(HEIGHT / 2), z_plane), u(100),		// DEAD CENTER
        //     0.0f, 0.0f, Surface_type::diffuse,
        //     RGB(0, 255, 0)
        // );

        u16 light_limit = (u16)scene.get_lights().size();
        u16 sphere_limit = (u16)scene.get_spheres().size();
        u16 bounce_limit = 5;

        scene.add_scene_detail(light_limit, sphere_limit, bounce_limit);
        scene.add_thread_group(16, 0, 0);

        scene_renderer.add_scene(&scene);
        scene_renderer.add_scene(&scene);
        scene_renderer.add_scene(&scene);
        scene_renderer.add_scene(&scene);
    }

    if (true)
    {
        {
            Scene scene;
            Setuper::setup_scene_0(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_1(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_2(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_3(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_4(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_5(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_6(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_7(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_8(&scene, "first");

            scene_renderer.add_scene(&scene);
        }

        {
            Scene scene;
            Setuper::setup_scene_9(&scene, "first");

            scene_renderer.add_scene(&scene);
        }
    }

    scene_renderer.combine_to_movie();

    return 0;
}
