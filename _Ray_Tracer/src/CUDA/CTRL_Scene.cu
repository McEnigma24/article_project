#include "CTRL_Scene.h"



Scene::Scene()
	:first_run(true), name(""), current_details_tab_index(0), current_thread_tab_index(0)
{
}
void Scene::assign_name(const string& _name)
{
	name = _name;
}
bool Scene::scene_empty()
{
	return ((light_points.size() == 0) || (spheres.size() == 0));
}
bool Scene::no_details_added()
{
	return ((scene_obj_specifics.size() == 0) || (scene_exe_specifics.size() == 0));
}
bool Scene::check_if_first_run()
{
	return (first_run);
}
string Scene::get_name()
{
	return name;
}
u64 Scene::how_many_details_scene()
{
	return scene_obj_specifics.size();
}
u64 Scene::how_many_details_exe()
{
	return scene_exe_specifics.size();
}
u16 Scene::get_details_exe_num_of_threads()
{
	return scene_exe_specifics[current_thread_tab_index].num_of_threads;
}
u16 Scene::get_details_exe_schema()
{
	return scene_exe_specifics[current_thread_tab_index].schema;
}
u16 Scene::get_details_exe_block_size()
{
	return scene_exe_specifics[current_thread_tab_index].block_size;
}
vector<Light_point>& Scene::get_lights()
{
	return light_points;
}
vector<Sphere>& Scene::get_spheres()
{
	return spheres;
}
vector<details>& Scene::get_details()
{
	return scene_obj_specifics;
}
Light_point* Scene::get_lights_ptr()
{
	return light_points.data();
}
Sphere* Scene::get_spheres_ptr()
{
	return spheres.data();
}
details* Scene::get_details_ptr()
{
	return scene_obj_specifics.data();
}

// TO POD KONIEC -> jezeli false -> to next_iteration()
bool Scene::is_SCENE_COMPLETED()
{
	#ifdef CPU
		if (scene_obj_specifics.size() == 1 && scene_exe_specifics.size() == 1)
		{
			return true;
		}

		return (
			(current_details_tab_index == scene_obj_specifics.size() - 1) &&
			(current_thread_tab_index == scene_exe_specifics.size() - 1)
			);
	#endif

	#ifdef GPU

		if (scene_obj_specifics.size() == 1)
			return true;

		return ( (current_details_tab_index == scene_obj_specifics.size() - 1) );

	#endif
}

void Scene::log_thread_details()
{
	linee("				next THREAD detail -> ");
	auto threads = scene_exe_specifics[current_thread_tab_index].num_of_threads;
	auto schema = scene_exe_specifics[current_thread_tab_index].schema;
	auto block_size = scene_exe_specifics[current_thread_tab_index].block_size;
	varr(threads);
	varr(schema);
	varr(block_size);
	nline;
}
void Scene::log_scene_details()
{
	linee("				next SCENE detail -> ");
	auto lights = scene_obj_specifics[current_details_tab_index].for_lights;
	auto spheres = scene_obj_specifics[current_details_tab_index].for_spheres;
	auto bounces = scene_obj_specifics[current_details_tab_index].for_bounces;
	varr(lights);
	varr(spheres);
	varr(bounces);
	nline;

	log_thread_details();
}

void Scene::next_settings()
{
	#ifdef CPU
		if (current_thread_tab_index + 1 <= scene_exe_specifics.size() - 1)
		{
			if (first_run) first_run = false;
			else
			{
				current_thread_tab_index++;
			}

			if(log_terminal_off) log_thread_details();
			return;
		}
	#endif

	if (current_details_tab_index + 1 <= scene_obj_specifics.size() - 1)
	{
		if (first_run) first_run = false;
		current_details_tab_index++;
		current_thread_tab_index = 0;
		
		if(log_terminal_off) log_scene_details();
		return;
	}
}

void Scene::add_scene_detail(u16 lights, u16 sph, u16 bounces)
{
	scene_obj_specifics.emplace_back
	(
		std::min(lights, static_cast<u16>(light_points.size())),
		std::min(sph, static_cast<u16>(spheres.size())),
		bounces
	);
}

void Scene::add_thread_group(u16 num_of_thr, u16 schema, u16 block_size)
{
	scene_exe_specifics.emplace_back(num_of_thr, schema, block_size);
}



void Scene::add_light(const d3& pos, const RGB& light_color)
{
	light_points.emplace_back(pos, light_color);

	spheres.emplace_back(pos, u(10), u(0), u(0), true, RGB_float(light_color), Surface_type::diffuse);
}

void Scene::add_sphere(const d3& c_pos, const unit& rad,
						const unit& transparent, const unit& reflective,
						Surface_type s_type, const RGB& col)
{	
	spheres.emplace_back(c_pos, rad, transparent, reflective, false, RGB_float(col), s_type);
}