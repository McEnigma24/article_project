#pragma once
#include "CTRL_STATS_Stat_Record.h"
#include "base/_preprocessor_.h"
#include <list>

class Scene;

class Scene_Controller
{
    u16 scene_index;
    vector<Scene> scenes;

    list<Stat_Record> all_gathered_stats;

    void setup_scenes();
    void next_scene();
    void add_current_stats_to_list();
    void calculate_progress_all();

public:
    Scene_Controller(u16 how_many_scenes);
    ~Scene_Controller();

    void choose_first_scene(u16 index);

    bool next_iteration();
};