#include "RT_Renderer.h"
#include "CTRL_Bmp.h"
#include "CTRL_Setuper.h"
#include "CTRL_Timer.h"
#include "FUNC_Float_functions.h"
#include "RT_Distance_contest_spheres.h"
#include "RT_RGB_float_sum_and_avg.h"
#include <cstdlib>

Renderer::Renderer(WIN_LINE(sf::RenderWindow& w))
    : WIN_LINE(window(w), sfml_pixels(sf::Points, G::PIXEL_ARRAY_SIZE), ) CPU_LINE(parallel_controller(this), )

          my_pixel(G::PIXEL_ARRAY_SIZE),
      scene_controller(SCENE_MAX_NUM)
{
    WIN_LINE(setup_all_pixels();)

    choose_first_scene(0); // scene_controller
}

bool Renderer::test_is_finished() { return !scene_controller.next_iteration(); }
void Renderer::choose_first_scene(u16 index) { scene_controller.choose_first_scene(index); }

#define CCE(x)                                                                                                         \
    {                                                                                                                  \
        cudaError_t err = x;                                                                                           \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            const string error = "CUDA ERROR - " + std::to_string(__LINE__) + " : " + __FILE__ + "\n";                 \
            cout << error;                                                                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

#ifdef GPU
__global__ void render_kernel(Renderer* dev_Renderer, RGB* dev_My_Pixels, Scene* dev_Current_Scene,
                              Light_point* dev_Current_Scene_lights, Sphere* dev_Current_Scene_spheres,
                              details* dev_Current_Scene_details)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(i < def_PIXEL_ARRAY_SIZE))
        return;

    dev_Renderer->per_pixel(i, dev_Current_Scene, dev_My_Pixels, dev_Current_Scene_lights, dev_Current_Scene_spheres,
                            dev_Current_Scene_details);
}
#endif

float calculate_progress()
{
    G::SCALING_MULTI;
    G::SCALING_ADD;

    float ret = (((float)G::PROGRESS_COUNTER) / ((float)G::PROGRESS_ALL));

    if (!((G::SCALING_MULTI == -1) && (G::SCALING_ADD == -1)))
    {
        ret = ret * G::SCALING_MULTI + G::SCALING_ADD;
    }
    else
    {
        // DEFAULT //
        ret = ret * 1.0f + 0.0f;
    }

    return ret;
}

string get_current_local_time_Ray_Tracer()
{
    ostringstream oss;

    if (G::REP_NUMBER == 1)
    {
        auto now = std::chrono::system_clock::now();
        std::time_t czas = std::chrono::system_clock::to_time_t(now);

        std::tm* aktualnyCzas = std::localtime(&czas);

        oss << std::put_time(aktualnyCzas, "%H:%M:%S") << std::endl;
    }
    else
    {
        auto now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* current_tm = std::localtime(&current_time);

        std::ifstream file("../_run_time_config_/estimated_finish_date.txt");
        if (!file.is_open())
        {
            std::cerr << "Nie mo?na otworzy? pliku: "
                      << "../_run_time_config_/estimated_finish_date.txt" << std::endl;
            exit(0);
        }

        int year, month, day, hour, minute, second; // 2024 11 15 10 48 02
        file >> year >> month >> day >> hour >> minute >> second;
        file.close();

        std::tm target_tm = {};
        target_tm.tm_year = year - 1900;
        target_tm.tm_mon = month - 1;
        target_tm.tm_mday = day;
        target_tm.tm_hour = hour;
        target_tm.tm_min = minute;
        target_tm.tm_sec = second;

        std::time_t target_time = std::mktime(&target_tm);
        if (target_time == -1)
        {
            std::cerr << "Nieprawid?owa data w pliku." << std::endl;
            return "error";
        }

        auto diff = std::difftime(target_time, current_time);
        if (diff < 0)
        {
            return "very close to finishing...\n";
        }

        auto diff_seconds = static_cast<int>(diff);
        int days = diff_seconds / (24 * 3600);
        diff_seconds %= (24 * 3600);
        int hours = diff_seconds / 3600;
        diff_seconds %= 3600;
        int minutes = diff_seconds / 60;
        int seconds = diff_seconds % 60;

        if (days != 0)
            oss << days << "d ";
        if (hours != 0)
            oss << hours << "h ";
        if (minutes != 0)
            oss << minutes << "m ";
        if (seconds != 0)
            oss << seconds << "s";
        oss << "\n";
    }

    return oss.str();
}

// MULTI THREADED
void Renderer::RENDER()
{
    //							 get's current scene (automatically)
    G::Render::current_scene_stats = new Stat_Record();

#ifdef CPU
    // SAME scene and details for couple rounds for avg stat
    parallel_controller.thread_group_host_and_round_keeper(
        G::REP_NUMBER, G::Render::current_scene->get_details_exe_num_of_threads(),
        G::Render::current_scene->get_details_exe_schema(), G::Render::current_scene->get_details_exe_block_size(),
        (G::Render::current_scene_stats->get_stats()));
#endif

#ifdef GPU

    // MAIN //

    int BLOCK_SIZE = def_BLOCK_SIZE;
    int NUMBER_OF_BLOCKS =
        def_PIXEL_ARRAY_SIZE / BLOCK_SIZE + 1; // zeby pokrywalo wszystko -> przez to musi byc warunek
    size_t size;

    size = sizeof(*this);
    Renderer* dev_Renderer{};
    CCE(cudaMalloc((void**)&dev_Renderer, size));
    CCE(cudaMemcpy(dev_Renderer, this, size, cudaMemcpyHostToDevice));

    size = sizeof(RGB) * def_PIXEL_ARRAY_SIZE;
    RGB* dev_My_Pixels{};
    CCE(cudaMalloc((void**)&dev_My_Pixels, size));
    CCE(cudaMemcpy(dev_My_Pixels, get_my_pixel(), size, cudaMemcpyHostToDevice));

    size = sizeof(*(G::Render::current_scene));
    Scene* dev_Current_Scene{};
    CCE(cudaMalloc((void**)&dev_Current_Scene, size));
    CCE(cudaMemcpy(dev_Current_Scene, G::Render::current_scene, size, cudaMemcpyHostToDevice));

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
            render_kernel<<<NUMBER_OF_BLOCKS, BLOCK_SIZE>>>(dev_Renderer, dev_My_Pixels, dev_Current_Scene,
                                                            dev_Current_Scene_lights, dev_Current_Scene_spheres,
                                                            dev_Current_Scene_details);
            CCE(cudaDeviceSynchronize());
        }
        kernel_timer.end();

        G::Render::current_scene_stats->get_stats().push_whole(kernel_timer.get_all_in_nano());
    }

    CCE(cudaMemcpy(get_my_pixel(), dev_My_Pixels, sizeof(RGB) * def_PIXEL_ARRAY_SIZE, cudaMemcpyDeviceToHost));

    // zwalnia od razu wszystko
    // CCE(cudaDeviceReset());

    CCE(cudaFree(dev_Renderer));
    CCE(cudaFree(dev_My_Pixels));
    CCE(cudaFree(dev_Current_Scene));
    CCE(cudaFree(dev_Current_Scene_lights));
    CCE(cudaFree(dev_Current_Scene_spheres));
    CCE(cudaFree(dev_Current_Scene_details));

#endif

    // deleting Stat_Record after exiting this function in Main -> test_is_finished()

    printf("[%.0f%]", (100.0f * calculate_progress()));
    cout << "       " << get_current_local_time_Ray_Tracer();
    G::PROGRESS_COUNTER++;

    if (((G::SCALING_MULTI == -1.0f) && (G::SCALING_ADD == -1.0f)))
    {
        line("SAVING FRAME - only when starting without params to save benchmark time");
        WIN_LINE(show_frame());
        save_frame();
    }
}

void Renderer::save_frame()
{
    static u64 gen_frame_counter = 0;

    gen_frame_counter++;
    string file_name = "output/img/";

#ifdef CPU
    file_name += "cpu_";
    ARCH_X86_LINE(file_name += "x86_");
    ARCH_X64_LINE(file_name += "x64_");
#endif

#ifdef GPU
    file_name += "gpu_";
    file_name += "CUDA_";
#endif

    UNIT_FLOAT_LINE(file_name += "float_");
    UNIT_DOUBLE_LINE(file_name += "double_");

    file_name += to_string(gen_frame_counter);
    file_name += ".bmp";

    BMP_static::save(file_name, my_pixel);
}

#ifdef WIN
void Renderer::setup_all_pixels()
{
    for (u16 y{}; y < G::HEIGHT; y++)
        for (u16 x{}; x < G::WIDTH; x++)
        {
            Vertex& current = sfml_pixels[convert_2d_to_1d(x, y)];
            current.color = Color::Black;
            current.position.x = static_cast<unit>(x);
            current.position.y = static_cast<unit>(y);
        }
}
void Renderer::reset_all_pixels()
{
    for (u16 y{}; y < G::HEIGHT; y++)
        for (u16 x{}; x < G::WIDTH; x++)
        {
            u64 i = convert_2d_to_1d(x, y);

            sfml_pixels[i].color = Color::Black;
            my_pixel[i] = {0, 0, 0};
        }
}

void Renderer::copy_from_pixel_colors()
{
    for (u16 y{}; y < G::HEIGHT; y++)
        for (u16 x{}; x < G::WIDTH; x++)
        {
            u64 i = convert_2d_to_1d(x, y);

            sfml_pixels[i].color = my_pixel[i];
        }
}
void Renderer::show_frame()
{
    copy_from_pixel_colors();

    {
        window.draw(sfml_pixels);
    }

    reset_all_pixels();
}
void Renderer::set_pixel_hack_for_showing_schema(const u64& coord, const Color& color)
{
    sfml_pixels[coord].color = color;
}
#endif

// Korzysta caly czas z ( G::Render::current_scene )

GPU_LINE(__host__ __device__)
void coordinates_for_camera_point(Ray& camera_ray, const u64& x, const u64& y)
{
    d3 starting_point((def_CAM_POS_X), (def_CAM_POS_Y), (def_CAM_POS_Z));

    d3 end_point(u(x) * def_CAM_INC_X, u(y) * def_CAM_INC_Y, 0);

#ifdef INTERACTIVE_MODE
    {
        // modified by position
        starting_point.x += G::Camera::move_offset.x;
        starting_point.y += G::Camera::move_offset.y;
        starting_point.z += G::Camera::move_offset.z;

        end_point.x += G::Camera::move_offset.x;
        end_point.y += G::Camera::move_offset.y;
        end_point.z += G::Camera::move_offset.z;

        // modified by rotation
        end_point.rotate_left_right(def_CAM_POS, G::Camera::rotation_angles.x);
        end_point.rotate_up_down(def_CAM_POS, G::Camera::rotation_angles.y);
        end_point.rotate_twist(def_CAM_POS, G::Camera::rotation_angles.z);
    }
#endif

    Ray::modify_after_construction(camera_ray, starting_point, end_point);
}

void coordinates_for_parallel(Ray& camera_ray, const u64& x, const u64& y)
{
    d3 starting_point(u(x) * G::Camera::inc_x, u(y) * G::Camera::inc_y, 0);

#ifdef INTERACTIVE_MODE
    {
        // modified by position
        starting_point.x += G::Camera::move_offset.x;
        starting_point.y += G::Camera::move_offset.y;
        starting_point.z += G::Camera::move_offset.z;

        // modified by rotation
    }
#endif

    camera_ray.modify_after_construction(starting_point, u(0), u(0), u(1));
}

GPU_LINE(__host__ __device__)
void Renderer::per_pixel(const u64& coord, Scene* current_scene, RGB* my_pixel, Light_point* current_scene_lights,
                         Sphere* current_scene_spheres, details* current_scene_details)
{
    const u64 x = static_cast<u64>(static_cast<u64>(coord) % def_WIDTH);
    const u64 y = static_cast<u64>(static_cast<u64>(coord) / def_WIDTH);

    per_pixel(x, y, current_scene, my_pixel, current_scene_lights, current_scene_spheres, current_scene_details);
}

#define DEB_end                                                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].r = 0;                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].g = 0;                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].b = 255;                                                                      \
    return;
#define DEB_hit                                                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].r = 0;                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].g = 255;                                                                      \
    my_pixel[def_convert_2d_to_1d(x, y)].b = 0;                                                                        \
    return;
#define DEB_miss                                                                                                       \
    my_pixel[def_convert_2d_to_1d(x, y)].r = 255;                                                                      \
    my_pixel[def_convert_2d_to_1d(x, y)].g = 0;                                                                        \
    my_pixel[def_convert_2d_to_1d(x, y)].b = 0;                                                                        \
    return;

#define COLOR_ASSIGN_END                                                                                               \
    my_pixel[def_convert_2d_to_1d(x, y)] = end_color;                                                                  \
    return;

struct path_info_entry
{
    // record dla kazdego trafionego

    Surface_type before_next_hit__surface_type;

    RGB_float_sum_and_avg before_next_hit__summed_color_from_light_sources; // last one have only this one

    unit before_next_hit__intensity_by_angle;
};

//												 	   r     g     b
#define COL_red                                                                                                        \
    RGB_float::assign(ret, RGB_float(u(1), u(0), u(0)));                                                               \
    return;
#define COL_green                                                                                                      \
    RGB_float::assign(ret, RGB_float(u(0), u(1), u(0)));                                                               \
    return;
#define COL_blue                                                                                                       \
    RGB_float::assign(ret, RGB_float(u(0), u(0), u(1)));                                                               \
    return;
#define COL_white                                                                                                      \
    RGB_float::assign(ret, RGB_float(u(1), u(1), u(1)));                                                               \
    return;
#define COL_grey                                                                                                       \
    RGB_float::assign(ret, RGB_float(u(0.5), u(0.5), u(0.5)));                                                         \
    return;
#define COL_pink                                                                                                       \
    RGB_float::assign(ret, RGB_float(u(1), u(0), u(1)));                                                               \
    return;

class Path
{
    // najpierw wyznaczamy przeciecie, potem propagujemy od tylu kolory
public:
    path_info_entry tab[def_MAX_BOUNCES];
    int this_scene_max_bounces;
    int index_of_last_added_element;

    GPU_LINE(__host__ __device__)
    Path(u16 how_many_bounces) : this_scene_max_bounces(how_many_bounces) { index_of_last_added_element = -1; }

    GPU_LINE(__host__ __device__)
    static void init(Path& obj, u16 how_many_bounces)
    {
        obj.this_scene_max_bounces = how_many_bounces;
        obj.index_of_last_added_element = -1;
    }

    GPU_LINE(__host__ __device__)
    static void backward_propagate_adding_color(Path& obj, RGB_float& ret)
    {
        RGB_float_sum_and_avg tmp_sum_avg;

        // ONLY one element
        if (obj.index_of_last_added_element == 0)
        {
            if (obj.tab[0].before_next_hit__surface_type == Surface_type::diffuse)
            {
                RGB_float_sum_and_avg::return_RGB_float(obj.tab[0].before_next_hit__summed_color_from_light_sources,
                                                        ret);
                return;
            }

            if (obj.tab[0].before_next_hit__surface_type != Surface_type::diffuse)
            {
                ret = RGB_float();
                return;
            }
        }

        // STOPING DIFFUSE in first hit
        if (obj.tab[0].before_next_hit__surface_type == Surface_type::diffuse)
        {
            RGB_float_sum_and_avg::return_RGB_float(obj.tab[0].before_next_hit__summed_color_from_light_sources, ret);
            return;
        }

        RGB_float previous_color;

        // LAST OBJ
        bool last_obj_mirror_or_refractive = false;
        switch (obj.tab[obj.index_of_last_added_element].before_next_hit__surface_type)
        {
        case Surface_type::diffuse:
        {
            previous_color = obj.tab[obj.index_of_last_added_element]
                                 .before_next_hit__summed_color_from_light_sources.return_RGB_float();

            break;
        }

        case Surface_type::mirror:
        case Surface_type::pass_through:
        {
            last_obj_mirror_or_refractive = true;
        }
        default:
            break;
        }

        for (int i = obj.index_of_last_added_element - 1; i != -1; i--)
        {
            auto& current = obj.tab[i];

            // MATERIAL PROPERTIES - pomijamy kolor luster i przepuszczalnych
            //							wstawiamy w to miejsce kolor wczesniejszego obiektu

            switch (current.before_next_hit__surface_type)
            {
            case Surface_type::diffuse:
            {
                // mamy kolor z wczesniejszego i go dodajemy do sumy

                if (!last_obj_mirror_or_refractive)
                {
                    current.before_next_hit__summed_color_from_light_sources +=
                        (previous_color * current.before_next_hit__intensity_by_angle);
                }

                previous_color = current.before_next_hit__summed_color_from_light_sources.return_RGB_float();

                break;
            }

            case Surface_type::mirror:
            case Surface_type::pass_through:
            {
                break;
            }
            default:
                break;
            }
        }

        ret = previous_color;
        return;
    }
};

GPU_LINE(__host__ __device__)
void forward_add_hit(Path& obj, const RGB_float_sum_and_avg& sum_of_light_sources, const unit& angle_intensity,
                     const Surface_type& surface_type)
{
    obj.index_of_last_added_element++;

    auto& current = obj.tab[obj.index_of_last_added_element];

    current.before_next_hit__surface_type = surface_type;
    RGB_float_sum_and_avg::assign(current.before_next_hit__summed_color_from_light_sources, sum_of_light_sources);

    current.before_next_hit__intensity_by_angle = angle_intensity;
}

GPU_LINE(__host__ __device__)
void Renderer::per_pixel(const u64& x, const u64& y, Scene* current_scene, RGB* my_pixel,
                         Light_point* current_scene_lights, Sphere* current_scene_spheres,
                         details* current_scene_details)
{
    Ray ray;
    RGB_float end_color;
    Hit_sphere hit;
    hit.intersecting_with_object = false;

    // SETTING UP PRIMARY RAY //
    coordinates_for_camera_point(ray, x, y);
    // coordinates_for_parallel(ray, x, y);

    const u16 BOUNCE_LIMIT = current_scene->get_details_scene_bounces(GPU_LINE(current_scene_details));
    CPU_LINE(u16 thread_number = (current_scene)->get_details_exe_num_of_threads());
    Path path(BOUNCE_LIMIT);

    Path::init(path, BOUNCE_LIMIT);

    // Shooting Ray //
    for (u16 i = 0; i < BOUNCE_LIMIT; i++)
    {
        ray_looking_for_sphere(hit, ray, current_scene, current_scene_lights, current_scene_spheres,
                               current_scene_details);

        // NO HIT //
        if (!hit.intersecting_with_object)
        {
            if (i == 0)
            {
                end_color = RGB_float(); // background color
                COLOR_ASSIGN_END;
            }
            else
                break;
        }

        Sphere* hit_sphere = &current_scene_spheres[hit.sphere_index];
        Ray::modify_after_construction(hit.normal, hit_sphere->center_pos, hit.intersection_pos);

        // LIGHT SOURCE HIT //
        if (hit_sphere->ligth_source)
        {
            if (i == 0)
            {
                end_color = hit_sphere->color;
                COLOR_ASSIGN_END;
            }
            else
            {
                RGB_float_sum_and_avg one_light;
                one_light.add_color(hit_sphere->color);
                forward_add_hit(path, one_light, u(1), Surface_type::diffuse);
                break;
            }
        }

        // SETTING UP next_ray
        switch (hit_sphere->surface_type)
        {
        case Surface_type::diffuse:
        {
            // Ray::modify_after_construction_random_in_normal_hemisphere(ray, hit.intersection_pos, hit.normal);
            // break;
        }

        case Surface_type::mirror:
        {
            // next_ray.modify_after_constructionp_perfect_mirror(hit.intersection_pos, ray.d, hit.normal);

            Ray::modify_after_constructionp_perfect_mirror(ray, hit.intersection_pos, ray.d, hit.normal);

            break;
        }

        case Surface_type::pass_through:
        {
        }
        default:
            break;
        }

        RGB_float_sum_and_avg local_sum_of_light_sources;
        Light_calculations::finding_light_sources_average_color_weighted_with_angle_and_distance(
            local_sum_of_light_sources, hit, current_scene, current_scene_lights, current_scene_spheres,
            current_scene_details);

        forward_add_hit(path, local_sum_of_light_sources, Light_calculations::intensity_based_on_angle(ray, hit.normal),
                        hit_sphere->surface_type);
    }

    // END //

    Path::backward_propagate_adding_color(path, end_color);
    COLOR_ASSIGN_END;
}

GPU_LINE(__host__ __device__)
void Renderer::ray_looking_for_sphere(Hit_sphere& hit, const Ray& ray, Scene* current_scene,
                                      Light_point* current_scene_lights, Sphere* current_scene_spheres,
                                      details* current_scene_details)
{
    Hit_sphere tmp_info;
    Distance_contest_spheres distance_contest;
    Distance_contest_spheres::init(distance_contest);

    CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_NULL((current_scene))));

    u16 limit = (current_scene)->get_details_scene_spheres(GPU_LINE(current_scene_details));

    CPU_LINE(u16 thread_number = (current_scene)->get_details_exe_num_of_threads());
    u16 limit_lights = (current_scene)->get_details_scene_lights(GPU_LINE(current_scene_details));
    u16 light_count = 0;

    for (u16 i = 0; i < limit; i++)
    {
        CPU_LINE(auto& sphere = (current_scene)->get_spheres()[i];)
        GPU_LINE(auto& sphere = current_scene_spheres[i];)

        // SPHERE around LIGHT SOURCE is skipped after exceding limit
        if (sphere.ligth_source)
        {
            if (light_count < limit_lights)
            {
                light_count++;
            }
            else
                continue;
        }

        sphere.intersection(tmp_info, ray);

        if (tmp_info.intersecting_with_object)
        {
            tmp_info.add_sphere_index(i);
            tmp_info.add_normal__provide_sphere_center(sphere.center_pos);

            distance_contest.new_sphere(tmp_info);
        }
    }

    distance_contest.get_closest_sphere(hit);
}
