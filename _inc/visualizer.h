// MOVE lib
#include "movie.h"

// MY Ray Tracing lib
#include "CTRL_Scene.h"
#include "CTRL_Setuper.h"
#include "RT_Renderer.h"

// BMP lib
#include "Bmp.h"

class Movie_Maker_Controller
{
    u64 movie_WIDTH;
    u64 movie_HEIGHT;
    vector<uint8_t> frame_buffer;
    vector<vector<RGB>> saved_frames;

    u64 convert_2d_to_1d_func(u64 x, u64 y) const { return y * movie_WIDTH + x; }

    void fill_frame_buffer(const vector<RGB>& render_output)
    {
        memset(frame_buffer.data(), 0, 4 * movie_WIDTH * movie_HEIGHT);

        for (unsigned int y = 0; y < movie_HEIGHT; y++)
            for (unsigned int x = 0; x < movie_WIDTH; x++)
            {
                frame_buffer[4 * movie_WIDTH * y + 4 * x + 2] = render_output[convert_2d_to_1d_func(x, y)].get_r();
                frame_buffer[4 * movie_WIDTH * y + 4 * x + 1] = render_output[convert_2d_to_1d_func(x, y)].get_g();
                frame_buffer[4 * movie_WIDTH * y + 4 * x + 0] = render_output[convert_2d_to_1d_func(x, y)].get_b();
            }
    }

public:
    Movie_Maker_Controller(u64 _WIDTH, u64 _HEIGHT) : movie_WIDTH(_WIDTH), movie_HEIGHT(_HEIGHT), frame_buffer(4 * movie_WIDTH * movie_HEIGHT) {}

    u64 get_WIDTH() const { return movie_WIDTH; }
    u64 get_HEIGHT() const { return movie_HEIGHT; }

    void add_new_frame(const vector<RGB>& frame) { saved_frames.push_back(frame); }

    void combine_to_movie(const string& name, int frame_rate = 1)
    {
        MovieWriter movie_writer(name, movie_WIDTH, movie_HEIGHT, frame_rate);

        int how_many_added_frames{};
        for (int i = 0; i < saved_frames.size(); i++)
        {
            fill_frame_buffer(saved_frames[i]);

            if (0 == i) { how_many_added_frames = 26; }
            else if ((saved_frames.size() - 1) == i) { how_many_added_frames = 34; }
            else { how_many_added_frames = 5; }

            for (int x{}; x < how_many_added_frames; x++)
                movie_writer.addFrame(&frame_buffer[0]);
        }
    }

    void delete_all_collected_frames() { saved_frames.clear(); }
};

class INPUT_scene_OUTPUT_movie
{
    Renderer render;
    Movie_Maker_Controller maker;
    string name;
    int frame_rate;

public:
    INPUT_scene_OUTPUT_movie(u64 _WIDTH, u64 _HEIGHT, const string& _name, int _frame_rate = 1)
        : render(_WIDTH, _HEIGHT), maker(_WIDTH, _HEIGHT), name(_name), frame_rate(_frame_rate)
    {
        Setuper::setup_Global_Variables___and___Clear_Stats();
    }

    void add_scene(Scene& scene)
    {
        var(&scene);
        G::Render::current_scene = &scene;
        line("1");
        render.RENDER();
        line("2");

        static u64 frame_counter{};
        line("3");
        BMP_static::save("output/frame_" + to_string(frame_counter++) + ".bmp", (Bmp_RGB*)render.get_my_pixel_vec().data(), maker.get_WIDTH(),
                         maker.get_HEIGHT());
        line("4");

        maker.add_new_frame(render.get_my_pixel_vec());
        line("5");
    }

    void combine_to_movie() { maker.combine_to_movie(name, frame_rate); }
};