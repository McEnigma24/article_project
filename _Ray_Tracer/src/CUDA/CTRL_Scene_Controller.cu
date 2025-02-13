#include "CTRL_Scene_Controller.h"
#include "CTRL_Scene.h"
#include "CTRL_Setuper.h"


void Scene_Controller::setup_scenes()
{
    #define setup_scene_index(x, name) Setuper::setup_scene_##x(&(scenes[x]), name);

    setup_scene_index(0, "FIRST");
    setup_scene_index(1, "BOX_1_LIGHT");
    setup_scene_index(2, "BOX_1_LIGHT_Left_Right_Mirror");

    setup_scene_index(3, "BOX_2_LIGHTS");
    setup_scene_index(4, "BOX_2_LIGHTS_Left_Right_Mirror");

    setup_scene_index(5, "MANY_SMALL_THREE_BIG__0_perc_mirror");
    setup_scene_index(6, "MANY_SMALL_THREE_BIG__25_perc_mirror");
    setup_scene_index(7, "MANY_SMALL_THREE_BIG__50_perc_mirror");
    setup_scene_index(8, "MANY_SMALL_THREE_BIG__75_perc_mirror");
    setup_scene_index(9, "MANY_SMALL_THREE_BIG__100_perc_mirror");

    calculate_progress_all();
}

void Scene_Controller::calculate_progress_all()
{
    for(auto& scene : scenes)
    {
        G::PROGRESS_ALL += scene.how_many_details_scene() CPU_LINE(* scene.how_many_details_exe());
    }
}

void Scene_Controller::next_scene()
{
    G::Render::current_scene = &(scenes[++scene_index]);
}



Scene_Controller::Scene_Controller(u16 how_many_scenes)
    :scene_index(0), scenes(how_many_scenes), all_gathered_stats()
{
    setup_scenes();
}

Scene_Controller::~Scene_Controller()
{
    string file_output;

    line("Saving collected stats");

    for (auto& stat : all_gathered_stats)
    {
        stat.save(file_output);
    }

    string stat_destination = "output/";

    CPU_LINE(stat_destination += "CPU");
    GPU_LINE(stat_destination += "GPU");

    stat_destination += "_";
    CPU_LINE(stat_destination += G::MODEL_NAME);
    GPU_LINE(stat_destination += G::MODEL_NAME);

    #ifdef CPU

        stat_destination += "_";
        ARCH_X86_LINE(stat_destination += "x86");
        ARCH_X64_LINE(stat_destination += "x64");

    #endif

    stat_destination += "_";
    UNIT_FLOAT_LINE(stat_destination += "float");
    UNIT_DOUBLE_LINE(stat_destination += "double");

    stat_destination += "_SAVED_STAT.txt";

    OUTPUT_TO_FILE(stat_destination, file_output);
}

void Scene_Controller::add_current_stats_to_list()
{
    CPU_LINE(SAFETY_CHECK(ASSERT_ER_IF_NULL(G::Render::current_scene_stats)));

    all_gathered_stats.push_back(*G::Render::current_scene_stats);

    // CLEAN UP
    delete G::Render::current_scene_stats;
    G::Render::current_scene_stats = null;
}

void Scene_Controller::choose_first_scene(u16 index)
{
    G::Render::current_scene = &(scenes[index]);
}

bool Scene_Controller::next_iteration()
{
    if(log_terminal_off) { line("next_iteration()") };

    if (G::Render::current_scene->check_if_first_run())
    {
        if(log_terminal_off)
        {
            nline;
            var(G::Render::current_scene->get_name());
        }
    }
    CPU_LINE(else { )
        add_current_stats_to_list();
    CPU_LINE(});


    if (!G::Render::current_scene->is_SCENE_COMPLETED())
    {
        if(log_terminal_off) line("turning next setting");
        G::Render::current_scene->next_settings();

        return true;
    }
    else
    {
        if(log_terminal_off) line("completed");
        next_scene();

        // scene could be empty, cause there are more tab elements than Setupper scene functions
        return !G::Render::current_scene->scene_empty();
    }
}