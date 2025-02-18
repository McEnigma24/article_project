#include "__preprocessor__.h"
#include "__time_stamp__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "engine.h"
#include "openMP_test.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));

    // proporcje obszaru obliczeniowego
    // 10x * 4x * x

    // r = 10;
    // x = 100;
    //           -> 40'000 sfer

    Engine engine(u(50), u(50), u(50), // sim space params
                                       // 1024, 768,                     // render params
                  1000, 1000,    // render params
                  "movie.mp4", 5 // movie params
    );
    engine.start();

    // Computation_Box box;
    // time_stamp("Box created");
    // box.fill_space_with_spheres(u(100), u(100), u(100));
    // time_stamp("Box filled with spheres");

    // const int WIDTH = 1000;
    // const int HEIGHT = 1000;

    // // const int WIDTH = 1024;
    // // const int HEIGHT = 768;

    // INPUT_scene_OUTPUT_movie scene_renderer(WIDTH, HEIGHT, "movie.mp4", 5);

    // if (false) // Manuly added scene
    // {
    //     Scene scene;
    //     // Setuper::setup_scene_0(&scene, "first");

    //     unit z_plane = u(100);
    //     scene.assign_name("scene");

    //     if (true)
    //     {
    //         unit z_plane = u(100);

    //         scene.add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-5000.0)), RGB(255, 255, 255));

    //         scene.add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-150))
    //                         //, RGB(255, 0, 0));
    //                         ,
    //                         RGB(255, 255, 255));

    //         scene.add_sphere(d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane), u(100), // DEAD CENTER
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 0));

    //         scene.add_sphere(d3(u(G::WIDTH / 2 + u(100)), u(G::HEIGHT / 2), z_plane + u(100)), u(100), // DEAD CENTER
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 0));

    //         scene.add_sphere(d3(u(G::WIDTH / 2 + u(75)), u(G::HEIGHT / 2 - 200),
    //                             z_plane // little higher - shadow maker
    //                                 - u(125)
    //                             //- u(50)
    //                             ),
    //                          u(40), 0.0f, 0.0f, Surface_type::diffuse, RGB(175, 255, 0));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300, z_plane + u(-150)),
    //                          u(100), // bottom - right
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(0, 255, 255));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300 + 50, z_plane), u(150), // bottom - right
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(0, 0, 255));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) - 300, z_plane), u(200), // top	  - right
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(255, 0, 255));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) + 300, z_plane), u(100), // bottom - left
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(255, 255, 0));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane), u(100), // top	  -	left
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(255, 255, 255));

    //         scene.add_sphere(d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane - u(25)),
    //                          u(80), // top	  -	left
    //                          0.0f, 0.0f, Surface_type::diffuse, RGB(255, 0, 0));
    //     }

    //     // scene.add_light
    //     // (
    //     //     d3(u(WIDTH / 2), u(HEIGHT / 2), z_plane)
    //     //     ,RGB(255, 255, 255)
    //     // );

    //     // scene.add_sphere
    //     // (
    //     //     d3(u(WIDTH / 2), u(HEIGHT / 2), z_plane), u(100),		// DEAD CENTER
    //     //     0.0f, 0.0f, Surface_type::diffuse,
    //     //     RGB(0, 255, 0)
    //     // );

    //     u16 light_limit = (u16)scene.get_lights().size();
    //     u16 sphere_limit = (u16)scene.get_spheres().size();
    //     u16 bounce_limit = 5;

    //     scene.add_scene_detail(light_limit, sphere_limit, bounce_limit);
    //     scene.add_thread_group(16, 0, 0);

    //     scene_renderer.add_scene(&scene);
    //     scene_renderer.add_scene(&scene);
    //     scene_renderer.add_scene(&scene);
    //     scene_renderer.add_scene(&scene);
    // }

    // if (false) // all Ray Tracing Scenes
    // {
    //     {
    //         Scene scene;
    //         Setuper::setup_scene_0(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_1(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_2(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_3(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_4(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_5(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_6(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_7(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_8(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }

    //     {
    //         Scene scene;
    //         Setuper::setup_scene_9(&scene, "first");

    //         scene_renderer.add_scene(&scene);
    //     }
    // }

    // if(true)
    // {
    //     Scene scene;
    //     box.

    // }

    // scene_renderer.combine_to_movie();

    return 0;
}
