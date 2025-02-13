#include "CTRL_Setuper.h"
#include "FUNC_Random_functions.h"
#include "CTRL_STATS_Stat_Record.h"
#include "CTRL_Parallel_CPU.h"

u64 G::WIDTH = 0xFFFFFFFFFFFFFFFF;
u64 G::HEIGHT = 0xFFFFFFFFFFFFFFFF;
u64 G::PIXEL_ARRAY_SIZE = 0xFFFFFFFFFFFFFFFF;

G::ray_type G::RAY_TYPE = G::camera_point;
u16 G::PIXEL_SKIP = 0xFFFF;

string G::MODEL_NAME = "default";
string G::NUM_OF_CORES = "-1";

u64 G::REP_NUMBER = 0;

float G::SCALING_MULTI = -1.0f;
float G::SCALING_ADD = -1.0f;

u64 G::PROGRESS_COUNTER = 0;
u64 G::PROGRESS_ALL = 0;

d3 G::Camera::pos = -1;
d3 G::Camera::starting_ray = -1;

d3 G::Camera::move_offset = -1;
unit G::Camera::move_speed = -1;

d3 G::Camera::rotation_angles = -1;
unit G::Camera::rotation_speed = -1;

unit G::Camera::pow2_pos_sum = -1;
unit G::Camera::inc_x = -1;
unit G::Camera::inc_y = -1;

Scene* G::Render::current_scene = nullptr;
Stat_Record* G::Render::current_scene_stats = nullptr;



void Setuper::setup_Global_Variables___and___Clear_Stats()
{
	G::WIDTH = 1000;	G::HEIGHT = 1000;
	//G::WIDTH = 1920;	G::HEIGHT = 1080;
	G::PIXEL_ARRAY_SIZE = (G::WIDTH * G::HEIGHT);

	G::RAY_TYPE = G::ray_type::camera_point;
	//G::RAY_TYPE = G::ray_type::parallel;

	G::PIXEL_SKIP = 1;


	// CAMERA
	G::Camera::pos = d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(
						//-1000000.0		// 1'000'000
						//-100000.0			// 100'000
		-10000.0			// 10'000
		//-1000.0			// 1'000
		//-100.0
		//-10.0
		//-1.0
		//0.0
		//1.0
		//10.0
	));

	G::Camera::pow2_pos_sum = pow2(G::Camera::pos.x) + pow2(G::Camera::pos.y) + pow2(G::Camera::pos.z);

	G::Camera::starting_ray = d3(u(0.0), u(0.0), u(0.0));

	G::Camera::move_offset = d3(u(0.0), u(0.0), u(0.0));
	G::Camera::move_speed = u(50);

	G::Camera::rotation_angles = d3(u(0.0), u(0.0), u(0.0));
	G::Camera::rotation_speed = u(M_PI / 1000);

	G::Camera::inc_x = u(1.0);
	G::Camera::inc_y = u(1.0);


	// RENDER
	G::Render::current_scene = nullptr;
}

// varr(name); var(scene);
#define setup_START(name) scene->assign_name(name);

#define setup_DEFAULT_details setup_default_scene_settings(scene);

#define setup_END ASSERT_ER_IF_TRUE(scene->no_details_added()); ASSERT_ER_IF_TRUE(scene->scene_empty());



void setup_default_scene_settings(Scene* scene)
{
	u16 light_limit = (u16)scene->get_lights().size();
	u16 sphere_limit = (u16)scene->get_spheres().size();
	u16 bounce_limit = def_DEFAULT_BOUNCE_LIMIT;

	scene->add_scene_detail(light_limit, sphere_limit, bounce_limit);
	// scene->add_scene_detail(light_limit, 3, bounce_limit);
	// scene->add_scene_detail(light_limit, 5, bounce_limit);

	//scene->add_thread_group(num_of_threads, Parallel_CPU::PR_solid_blocks, 1);
	//scene->add_thread_group(num_of_threads, Parallel_CPU::PR_subsequent, 1);
	//scene->add_thread_group(num_of_threads, Parallel_CPU::PR_subsequent_custom, 100);

	for (u16 i = 1; i <= def_MAX_NUM_OF_THREADS; i++)
	{
		scene->add_thread_group(i, Parallel_CPU::PR_solid_blocks, 0);
		//scene->add_thread_group(i, Parallel_CPU::PR_subsequent, 0);
		//scene->add_thread_group(i, Parallel_CPU::PR_subsequent_custom, 550);
	}

	ASSERT_ER_IF_TRUE(scene->no_details_added());
}

// called at -> CTRL_Scene_Controller.cpp

void Setuper::setup_scene_0(Scene* scene, const string& name)
{
	setup_START(name);

	// Manual
	{
		// coords -> y is small at top
		
		// dedicated light pos
		{
			//d3 light_pos(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(-150));

			//scene->add_light(
			//					light_pos,
			//					RGB(255, 255, 255)
			//				);

			//scene->add_sphere(
			//					light_pos, u(10),	// bottom - right
			//					0.0f, 0.0f, Surface_type::diffuse,
			//					RGB(0, 255, 0)
			//				);
		}


		unit z_plane = u(100);

		//scene->add_sphere(
		//	d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane), u(100),		// DEAD CENTER
		//	0.0f, 0.0f, Surface_type::diffuse,
		//	RGB(0, 255, 0));



		//scene->add_light(
		//	d3(u(G::WIDTH / 2 - 150), u(G::HEIGHT / 2 - 150), (z_plane 
		//																//+ u(1000)
		//																//+ u(750)
		//																//+ u(500)
		//																//+ u(250)
		//																//+ u(100)
		//																- u(0)			// in plane
		//																//- u(100)																		
		//																//- u(250)
		//																//- u(500)
		//																//- u(750)
		//																//- u(1000)
		//														)),
		//	RGB(255, 255, 255)
		//	//RGB(255, 125, 255)
		//);

		
		// BLIKO KAMERY o�wietla wszystko
		scene->add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-5000.0))
			,RGB(255, 255, 255));

		scene->add_light(d3(u(G::WIDTH / 2 - 250 + 50), u(G::HEIGHT / 2 - 50), u(-150))
			//, RGB(255, 0, 0));
			, RGB(255, 255, 255));

		//scene->add_light(d3(u(G::WIDTH / 2 + 250), u(G::HEIGHT / 2), u(-5550))
		//	//, RGB(255, 0, 0));
		//	, RGB(255, 255, 255));

		//unit separation = u(0.5);													// MAIN - blurry shadow
		//unit light_source_side = u(25) * separation;
		//for (unit y{}; y < light_source_side; y += separation)
		//{
		//	for (unit x{}; x < light_source_side; x += separation)
		//	{
		//		scene->add_light(d3(msm.x + u(x), msm.y + u(y), msm.z + u(0))
		//			, RGB(255, 255, 255));
		//	}
		//}





		//scene->add_light(
		//	d3(u(G::WIDTH / 2 + 150), u(G::HEIGHT / 2 - 150), (z_plane - u(175))),	// SECOND LIGHT shadow brightener
		//	//RGB(0, 0, 255)
		//	RGB(255, 255, 255)
		//);




		//scene->add_light(
		//	d3(u(G::WIDTH / 2 - 300), u(G::HEIGHT / 2), z_plane),		// center - left
		//	RGB(255, 255, 255)
		//);

		//scene->add_light(
		//	d3(u(G::WIDTH / 2 + 300), u(G::HEIGHT / 2), z_plane),		// center - right
		//	RGB(255, 255, 255)
		//);

		//scene->add_light(
		//	d3(u(G::WIDTH / 2), u(G::HEIGHT / 2 + 300), z_plane),		// center - bottom
		//	RGB(255, 255, 255)
		//);

		//scene->add_light(
		//	d3(u(G::WIDTH / 2), u(G::HEIGHT / 2 - 300), z_plane),		// center - top
		//	RGB(255, 255, 255)
		//);



		scene->add_sphere(
			d3(u(G::WIDTH / 2), u(G::HEIGHT / 2), z_plane), u(100),		// DEAD CENTER
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(0, 255, 0));

		scene->add_sphere(
			d3(u(G::WIDTH / 2 + u(100)), u(G::HEIGHT / 2), z_plane + u(100)), u(100),		// DEAD CENTER
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(0, 255, 0));


		int rad = 2;
		for (int y = -rad; y < rad; y++)
		{
			for (int x = -rad; x < rad; x++)
			{
				//scene->add_sphere(
				//	d3(u((G::WIDTH / 2) + x * 300), u((G::HEIGHT / 2) + y * 300), z_plane + u(100)), u(50),		// DEAD CENTER
				//	0.0f, 0.0f, Surface_type::diffuse,
				//	RGB(0, 255, 0));
			}
		}


		scene->add_sphere(
			d3(u(G::WIDTH / 2 + u(75)), u(G::HEIGHT / 2 - 200), z_plane			// little higher - shadow maker
																		- u(125)
																		//- u(50)
			), u(40),
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(175, 255, 0));

		//scene->add_sphere(
		//	d3(u(G::WIDTH / 2), u(G::HEIGHT / 2 - 200), z_plane), u(50),	// little higher - previous
		//	0.0f, 0.0f, Surface_type::diffuse,
		//	RGB(175, 255, 0));

		scene->add_sphere(
			d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300, z_plane + u(-150)), u(100),	// bottom - right
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(0, 255, 255));

		scene->add_sphere(
			d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) + 300 + 50, z_plane), u(150),	// bottom - right
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(0, 0, 255));
	
		scene->add_sphere(
			d3(u(G::WIDTH / 2) + 300, u(G::HEIGHT / 2) - 300, z_plane), u(200),	// top	  - right
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(255, 0, 255));

		scene->add_sphere(
			d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) + 300, z_plane), u(100),	// bottom - left
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(255, 255, 0));

		scene->add_sphere(
			d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane), u(100),	// top	  -	left
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(255, 255, 255));
		

		scene->add_sphere(
			d3(u(G::WIDTH / 2) - 300, u(G::HEIGHT / 2) - 300, z_plane - u(25)), u(80),	// top	  -	left
			0.0f, 0.0f, Surface_type::diffuse,
			RGB(255, 0, 0));
	}


	setup_DEFAULT_details;

	//scene->add_scene_detail(100, 1500, 2);

	setup_END;
}

// No mirror - 1 light
void Setuper::setup_scene_1(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));

	// 1 o�wietlaj�ce wszystkie
	{
		scene->add_light(d3(center.x, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
	}

	// "BOX"
	{
		unit rad = u(1000);
		// LEFT side														SECONDARY is blocked by this
		scene->add_sphere(
			d3(0 - rad + u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0),
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(255, 0, 0)
		);

		// RIGHT side
		scene->add_sphere(
			d3(G::WIDTH + rad - u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0),
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(0, 255, 0)
		);

		// UP side															SECONDARY ray hits this
		scene->add_sphere(
			d3(center.x, 0 - rad + u(50), center.z), rad * 1.4f,
			u(0), u(0),
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(0, 0, 255)
		);

		// DOWN side
		scene->add_sphere(
			d3(center.x, G::HEIGHT + rad - u(50), center.z), rad * 1.4f,
			u(0), u(0),
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(255, 140, 0)
		);

		// BACKGROUND
		scene->add_sphere(
			d3(center.x, center.y, center.z + rad + u(50)), rad * 1.4f,
			u(0), u(0),
			//Surface_type::mirror,
			Surface_type::diffuse,
			//RGB(150, 0, 150)
			RGB(230, 230, 230)
		);
	}

	// spheres inside
	{
		unit rad = u(50);
		unit z_plane = u(-800);

		scene->add_sphere(
			d3(center.x - u(2.3) * rad, center.y, z_plane), rad
			, u(0), u(0), Surface_type::diffuse,
			RGB(0, 0, 250)
		);

		scene->add_sphere(
			d3(center.x, center.y, z_plane), rad
			, u(0), u(0),
			Surface_type::diffuse,
			//Surface_type::mirror,
			RGB(0, 250, 0)
		);

		scene->add_sphere(
			d3(center.x + u(2.3) * rad, center.y, z_plane), rad
			, u(0), u(0), Surface_type::diffuse,
			RGB(250, 0, 0)
		);
	}

	setup_DEFAULT_details;

	setup_END;
}

// Left Right mirror - 1 light
void Setuper::setup_scene_2(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	
	// 1 o�wietlaj�ce wszystkie
	{
		scene->add_light(d3(center.x, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
	}

	// 1 nad sferami - NON active
	{
		//scene->add_light(d3(center.x, center.y - 75, center.z - 500)
		//	, RGB(255, 255, 255));
	}


	// "BOX"
	{
		unit rad = u(1000);
		// LEFT side														SECONDARY is blocked by this
		scene->add_sphere(
			d3(0 - rad + u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), 
			Surface_type::mirror,
			//Surface_type::diffuse,
			RGB(255, 0, 0)
		);

		// RIGHT side
		scene->add_sphere(
			d3(G::WIDTH + rad - u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), 
			Surface_type::mirror,
			//Surface_type::diffuse,
			RGB(0, 255, 0)
		);

		// UP side															SECONDARY ray hits this
		scene->add_sphere(
			d3(center.x, 0 - rad + u(50), center.z), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(0, 0, 255)
		);

		// DOWN side
		scene->add_sphere(
			d3(center.x, G::HEIGHT + rad - u(50), center.z), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(255, 140, 0)
		);

		// BACKGROUND
		scene->add_sphere(
			d3(center.x, center.y, center.z + rad + u(50)), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			//RGB(150, 0, 150)
			RGB(230, 230, 230)
		);
	}

	// spheres inside
	{
		unit rad = u(50);
		unit z_plane = u(-800);

		scene->add_sphere(
			d3(center.x - u(2.3) * rad, center.y, z_plane), rad
			,u(0), u(0), Surface_type::diffuse,
			RGB(0, 0, 250)
		);

		scene->add_sphere(
			d3(center.x, center.y, z_plane), rad
			,u(0), u(0), 
			Surface_type::diffuse,
			//Surface_type::mirror,
			RGB(0, 250, 0)
		);

		scene->add_sphere(
			d3(center.x + u(2.3) * rad, center.y, z_plane), rad
			,u(0), u(0), Surface_type::diffuse,
			RGB(250, 0, 0)
		);
	}

	setup_DEFAULT_details;

	setup_END;
}

// No mirror - 2 light
void Setuper::setup_scene_3(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));

	// 3 o�wietlaj�ce wszystkie
	{
		unit diviadion = u(250);

		// // 1 za nimi 
		// scene->add_light(d3(center.x, center.y - 375, center.z - 1500)
		// 	, RGB(255, 255, 255));

		// scene->add_light(d3(center.x, center.y - 275, center.z - 2000)
		// 	, RGB(255, 255, 255));
		// 	//, RGB(255, 0, 0));

		scene->add_light(d3(center.x - diviadion, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
			//, RGB(0, 255, 0));

		scene->add_light(d3(center.x + diviadion, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
			//, RGB(0, 0, 255));
	}

	// "BOX"
	{
		unit rad = u(1000);
		// LEFT side
		scene->add_sphere(
			d3(0 - rad + u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), Surface_type::diffuse,
			RGB(255, 0, 0)
		);

		// RIGHT side
		scene->add_sphere(
			d3(G::WIDTH + rad - u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), Surface_type::diffuse,
			RGB(0, 255, 0)
		);

		// UP side
		scene->add_sphere(
			d3(center.x, 0 - rad + u(50), center.z), rad * 1.4f,
			u(0), u(0), Surface_type::diffuse,
			RGB(0, 0, 255)
		);

		// DOWN side
		scene->add_sphere(
			d3(center.x, G::HEIGHT + rad - u(50), center.z), rad * 1.4f,
			u(0), u(0), Surface_type::diffuse,
			RGB(255, 140, 0)
		);

		// BACKGROUND
		scene->add_sphere(
			d3(center.x, center.y, center.z + rad + u(50)), rad * 1.4f,
			u(0), u(0), Surface_type::diffuse,
			//RGB(150, 0, 150)
			RGB(230, 230, 230)
		);
	}

	// spheres inside
	{
		unit rad = u(50);
		unit z_plane = u(-800);

		scene->add_sphere(
			d3(center.x - u(2.3) * rad, center.y, z_plane), rad
			, u(0), u(0), Surface_type::diffuse,
			RGB(0, 0, 255)
		);

		scene->add_sphere(
			d3(center.x, center.y, z_plane), rad
			, u(0), u(0), Surface_type::diffuse,
			RGB(0, 255, 0)
		);

		scene->add_sphere(
			d3(center.x + u(2.3) * rad, center.y, z_plane), rad
			, u(0), u(0), Surface_type::diffuse,
			RGB(255, 0, 0)
		);
	}


	setup_DEFAULT_details;

	//scene->add_scene_detail(1, 4, 1);
	//scene->add_scene_detail(1, 6, 1);
	//scene->add_scene_detail(1, 8, 1);
	//scene->add_scene_detail(1, 11, 1);
	//scene->add_scene_detail(2, 11, 1);
	//scene->add_scene_detail(3, 11, 1);
	//scene->add_scene_detail(3, 11, 2);
	//scene->add_scene_detail(3, 11, 3);

	//scene->add_thread_group(1, 0, 1);

	setup_END;
}

// Left Right mirror - 2 light
void Setuper::setup_scene_4(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));

	// 3 o�wietlaj�ce wszystkie
	{
		unit diviadion = u(250);

		// // 1 za nimi 
		// scene->add_light(d3(center.x, center.y - 375, center.z - 1500)
		// 	, RGB(255, 255, 255));

		// scene->add_light(d3(center.x, center.y - 275, center.z - 2000)
		// 	, RGB(255, 255, 255));
		// 	//, RGB(255, 0, 0));

		scene->add_light(d3(center.x - diviadion, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
			//, RGB(0, 255, 0));

		scene->add_light(d3(center.x + diviadion, center.y - 275, center.z - 2000)
			, RGB(255, 255, 255));
			//, RGB(0, 0, 255));
	}

	// "BOX"
	{
		unit rad = u(1000);
		// LEFT side														SECONDARY is blocked by this
		scene->add_sphere(
			d3(0 - rad + u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), 
			Surface_type::mirror,
			//Surface_type::diffuse,
			RGB(255, 0, 0)
		);

		// RIGHT side
		scene->add_sphere(
			d3(G::WIDTH + rad - u(50), center.y, center.z), rad * 1.4f,
			u(0), u(0), 
			Surface_type::mirror,
			//Surface_type::diffuse,
			RGB(0, 255, 0)
		);

		// UP side															SECONDARY ray hits this
		scene->add_sphere(
			d3(center.x, 0 - rad + u(50), center.z), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(0, 0, 255)
		);

		// DOWN side
		scene->add_sphere(
			d3(center.x, G::HEIGHT + rad - u(50), center.z), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			RGB(255, 140, 0)
		);

		// BACKGROUND
		scene->add_sphere(
			d3(center.x, center.y, center.z + rad + u(50)), rad * 1.4f,
			u(0), u(0), 
			//Surface_type::mirror,
			Surface_type::diffuse,
			//RGB(150, 0, 150)
			RGB(230, 230, 230)
		);
	}

	// spheres inside
	{
		unit rad = u(50);
		unit z_plane = u(-800);

		scene->add_sphere(
			d3(center.x - u(2.3) * rad, center.y, z_plane), rad
			,u(0), u(0), Surface_type::diffuse,
			RGB(0, 0, 250)
		);

		scene->add_sphere(
			d3(center.x, center.y, z_plane), rad
			,u(0), u(0), 
			Surface_type::diffuse,
			//Surface_type::mirror,
			RGB(0, 250, 0)
		);

		scene->add_sphere(
			d3(center.x + u(2.3) * rad, center.y, z_plane), rad
			,u(0), u(0), Surface_type::diffuse,
			RGB(250, 0, 0)
		);
	}

	setup_DEFAULT_details;

	setup_END;
}

// all diffuse
void Setuper::setup_scene_5(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	unit rad = u(10000);
	unit small_rad = u(50);
	d3 ground(center.x, G::HEIGHT + u(0.92) * rad, center.z + rad);

	// 1 o�wietlaj�ce wszystkie
	{
		// G��WNE
		scene->add_light(
			d3(center.x, G::HEIGHT - u(1500), center.z + u(6000))
			, RGB(255, 255, 255));
	}

	// POD�O�E
	{	
		scene->add_sphere(
			ground, rad,
			u(0), u(0), 
			Surface_type::diffuse,
			//Surface_type::mirror,
			//RGB(255, 140, 0)
			RGB(109, 114, 122)
		);
	}

	// KILKA MA�YCH
	{
		unit y = G::HEIGHT + u(-1000) + u(450);

		// 3 na g�rze
		{
			//// Du�a
			scene->add_sphere(
				d3(center.x - u(350), y, center.z + rad - u(3000)), u(200),
				u(0), u(0), 
				Surface_type::diffuse,
				//Surface_type::mirror,
				RGB(255, 0, 123)
			);

			//// �rednia
			scene->add_sphere(
				d3(center.x, y + u(50), center.z + rad - u(3000)), u(150),
				u(0), u(0), Surface_type::diffuse,
				RGB(0, 255, 123)
			);

			//// Ma�a
			scene->add_sphere(
				d3(center.x + u(260), y + (88), center.z + rad - u(3000)), u(112),
				u(0), u(0), Surface_type::diffuse,
				RGB(123, 0, 255)
			);
		}

		// wcze�niej wylosowane teraz na sta�e
		{
			#define one_small_sphere(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::diffuse, \
				RGB(r, g, b) \
				); \
			}

			// uglier one
			if (0)
			{
				one_small_sphere(86, 245, 238, 386, 1120, 5692);
				one_small_sphere(18, 5, 135, 931, 760, 6576);
				one_small_sphere(33, 83, 204, 1001, 424, 7724);
				one_small_sphere(175, 248, 176, -203, 256, 8732);
				one_small_sphere(188, 138, 53, 995, 1364, 5235);
				one_small_sphere(16, 212, 138, 696, 1015, 5924);
				one_small_sphere(232, 45, 149, 20, 1215, 5521);
				one_small_sphere(203, 145, 177, 853, 456, 7566);
				one_small_sphere(136, 109, 160, -8, 350, 8071);
				one_small_sphere(193, 135, 57, 1002, 353, 8058);
				one_small_sphere(120, 46, 194, 529, 1285, 5358);
				one_small_sphere(96, 132, 24, 1054, 317, 8266);
				one_small_sphere(117, 95, 235, 448, 580, 7095);
				one_small_sphere(219, 151, 178, 873, 1187, 5568);
				one_small_sphere(64, 143, 54, -179, 898, 6253);
				one_small_sphere(226, 239, 213, 1357, 268, 8730);
				one_small_sphere(28, 210, 233, 1265, 433, 7760);
				one_small_sphere(232, 207, 232, 360, 1288, 5354);
				one_small_sphere(246, 204, 44, -105, 953, 6108);
				one_small_sphere(88, 175, 240, 826, 1048, 5857);
				one_small_sphere(47, 220, 44, 1155, 736, 6678);
				one_small_sphere(90, 92, 156, 1337, 737, 6717);
				one_small_sphere(160, 173, 82, 766, 282, 8401);
				one_small_sphere(156, 48, 140, -219, 1109, 5775);
				one_small_sphere(96, 25, 246, -354, 1249, 5508);
				one_small_sphere(169, 4, 165, -311, 830, 6454);
				one_small_sphere(227, 0, 203, 53, 337, 8125);
				one_small_sphere(110, 57, 179, 728, 549, 7209);
				one_small_sphere(126, 36, 137, 917, 393, 7845);
				one_small_sphere(126, 100, 228, 971, 588, 7104);
				one_small_sphere(211, 91, 184, 962, 422, 7726);
				one_small_sphere(90, 5, 83, -306, 1075, 5864);
				one_small_sphere(31, 17, 107, 276, 593, 7059);
				one_small_sphere(180, 177, 138, 214, 1320, 5300);
				one_small_sphere(171, 104, 14, 1119, 436, 7705);
				one_small_sphere(120, 122, 169, 943, 1076, 5808);
				one_small_sphere(94, 137, 160, -475, 1264, 5503);
				one_small_sphere(171, 228, 135, 738, 758, 6563);
				one_small_sphere(163, 242, 82, -209, 884, 6295);
				one_small_sphere(31, 91, 195, -178, 358, 8083);
				one_small_sphere(35, 227, 22, -346, 295, 8531);
				one_small_sphere(54, 217, 103, 576, 1147, 5634);
				one_small_sphere(101, 199, 153, -478, 346, 8286);
				one_small_sphere(223, 183, 41, -504, 545, 7399);
				one_small_sphere(84, 232, 165, 33, 1032, 5908);
				one_small_sphere(171, 185, 79, 1161, 316, 8310);
				one_small_sphere(191, 46, 210, 151, 1224, 5491);
				one_small_sphere(2, 10, 92, 142, 1256, 5429);
				one_small_sphere(129, 167, 230, 1371, 269, 8731);
				one_small_sphere(210, 197, 121, -445, 987, 6096);
				one_small_sphere(163, 230, 173, 166, 1163, 5614);
				one_small_sphere(218, 179, 224, 1070, 1017, 5954);
				one_small_sphere(228, 116, 204, -51, 1226, 5508);
				one_small_sphere(183, 218, 214, 1240, 538, 7337);
				one_small_sphere(0, 151, 199, -336, 599, 7150);
				one_small_sphere(31, 212, 78, -367, 1142, 5731);
				one_small_sphere(185, 128, 183, -231, 615, 7064);
				one_small_sphere(175, 46, 179, 1168, 734, 6689);
				one_small_sphere(151, 9, 209, 818, 471, 7503);
				one_small_sphere(164, 161, 56, 709, 227, 8780);
				one_small_sphere(221, 163, 245, -268, 1146, 5704);
				one_small_sphere(132, 52, 149, -218, 892, 6275);
				one_small_sphere(99, 7, 232, -351, 588, 7193);
				one_small_sphere(33, 194, 231, -316, 269, 8698);
				one_small_sphere(164, 76, 103, 1005, 476, 7515);
				one_small_sphere(21, 47, 150, 883, 1319, 5309);
				one_small_sphere(4, 112, 80, 131, 619, 6986);
				one_small_sphere(214, 217, 183, -136, 350, 8114);
				one_small_sphere(106, 166, 52, 273, 957, 6058);
				one_small_sphere(110, 190, 136, -339, 1392, 5231);
				one_small_sphere(246, 40, 64, 279, 1125, 5686);
				one_small_sphere(91, 57, 4, 13, 522, 7336);
				one_small_sphere(179, 106, 202, 220, 1148, 5640);
				one_small_sphere(152, 144, 155, 313, 855, 6305);
				one_small_sphere(127, 119, 136, 245, 495, 7404);
				one_small_sphere(67, 103, 7, 1178, 955, 6115);
				one_small_sphere(251, 115, 159, 1084, 615, 7031);
				one_small_sphere(169, 111, 218, 304, 267, 8490);
				one_small_sphere(120, 159, 35, 154, 456, 7564);
				one_small_sphere(243, 118, 20, 329, 1001, 5953);
				one_small_sphere(251, 132, 92, 307, 660, 6841);
				one_small_sphere(87, 93, 41, 1183, 1129, 5725);
				one_small_sphere(5, 43, 14, 1081, 1375, 5224);
				one_small_sphere(103, 223, 124, -68, 376, 7959);
				one_small_sphere(253, 91, 116, -229, 422, 7796);
				one_small_sphere(10, 120, 167, 88, 286, 8651);
			}

			// #3
			if (0)
			{
				one_small_sphere(189, 243, 127, 421, 389, 78230);
				one_small_sphere(52, 158, 69, 1281, 530, 7378);
				one_small_sphere(48, 222, 75, 698, 417, 7709);
				one_small_sphere(31, 223, 251, -207, 905, 6242);
				one_small_sphere(101, 195, 104, 591, 1034, 5877);
				one_small_sphere(248, 42, 119, 441, 1313, 5305);
				one_small_sphere(139, 67, 59, 1165, 267, 8625);
				one_small_sphere(126, 101, 244, 1052, 584, 7132);
				one_small_sphere(57, 68, 174, 1369, 283, 8624);
				one_small_sphere(144, 43, 137, -488, 575, 7283);
				one_small_sphere(33, 93, 19, 1314, 1251, 5497);
				one_small_sphere(35, 219, 30, 529, 890, 6212);
				one_small_sphere(228, 32, 155, -245, 284, 8545);
				one_small_sphere(170, 208, 39, 1064, 709, 6740);
				one_small_sphere(235, 16, 184, 9, 406, 7802);
				one_small_sphere(109, 69, 2, 26, 302, 8330);
				one_small_sphere(25, 9, 85, 705, 307, 8247);
				one_small_sphere(35, 84, 68, 199, 606, 7018);
				one_small_sphere(129, 150, 233, -507, 1149, 5748);
				one_small_sphere(92, 104, 148, -241, 254, 8771);
				one_small_sphere(84, 33, 92, 818, 506, 7370);
				one_small_sphere(159, 234, 246, 1150, 1110, 5760);
				one_small_sphere(18, 192, 244, 670, 315, 8199);
				one_small_sphere(111, 26, 36, 102, 911, 6183);
				one_small_sphere(188, 225, 253, 566, 335, 8086);
				one_small_sphere(127, 178, 183, 789, 655, 6865);
				one_small_sphere(76, 36, 61, 1149, 730, 6694);
				one_small_sphere(207, 38, 186, 591, 606, 7004);
				one_small_sphere(22, 39, 197, -49, 521, 7354);
				one_small_sphere(27, 16, 65, 255, 992, 5978);
				one_small_sphere(107, 162, 47, 1262, 484, 7548);
				one_small_sphere(114, 46, 2, 1224, 352, 8132);
				one_small_sphere(45, 184, 140, 346, 793, 6464);
				one_small_sphere(165, 127, 205, 1332, 620, 7075);
				one_small_sphere(212, 180, 230, -354, 568, 7262);
				one_small_sphere(6, 204, 106, 1088, 943, 6130);
				one_small_sphere(199, 129, 56, 1056, 305, 8335);
				one_small_sphere(47, 171, 157, -178, 735, 6686);
				one_small_sphere(127, 212, 41, 291, 537, 7249);
				one_small_sphere(225, 228, 20, 771, 258, 8562);
				one_small_sphere(241, 209, 53, 42, 292, 8383);
				one_small_sphere(124, 109, 201, 68, 236, 8768);
				one_small_sphere(155, 236, 133, -168, 1182, 5613);
				one_small_sphere(195, 72, 198, 199, 723, 6665);
				one_small_sphere(57, 204, 70, 265, 453, 7565);
				one_small_sphere(209, 248, 232, 625, 397, 7791);
				one_small_sphere(41, 52, 166, 1119, 419, 7777);
				one_small_sphere(160, 65, 246, 745, 304, 8266);
				one_small_sphere(45, 240, 71, 99, 538, 7264);
				one_small_sphere(222, 206, 22, 867, 837, 6365);
				one_small_sphere(112, 217, 51, 581, 633, 6920);
				one_small_sphere(198, 29, 153, -482, 344, 8298);
				one_small_sphere(177, 86, 94, -275, 446, 7707);
				one_small_sphere(99, 211, 74, 830, 334, 8116);
				one_small_sphere(30, 219, 74, 1044, 438, 7675);
				one_small_sphere(19, 180, 211, 746, 1361, 5221);
				one_small_sphere(136, 6, 201, 586, 237, 8690);
				one_small_sphere(183, 137, 28, 431, 453, 7553);
				one_small_sphere(226, 124, 251, 1333, 1369, 5273);
				one_small_sphere(148, 42, 64, 833, 276, 8450);
				one_small_sphere(7, 150, 79, 727, 406, 7761);
				one_small_sphere(91, 87, 92, 257, 510, 7349);
				one_small_sphere(209, 44, 57, -338, 410, 7893);
				one_small_sphere(180, 198, 83, 557, 1146, 5636);
				one_small_sphere(244, 235, 100, 1192, 361, 8075);
				one_small_sphere(175, 223, 41, 599, 922, 6137);
				one_small_sphere(133, 86, 49, 445, 237, 8686);
				one_small_sphere(96, 177, 165, 880, 574, 7139);
				one_small_sphere(129, 58, 159, 78, 512, 7361);
				one_small_sphere(249, 202, 90, 403, 252, 8578);
				one_small_sphere(22, 106, 143, 924, 1326, 5300);
				one_small_sphere(65, 254, 160, -125, 705, 6762);
				one_small_sphere(186, 190, 151, 31, 265, 8563);
				one_small_sphere(229, 6, 238, 874, 750, 6597);
				one_small_sphere(3, 20, 134, 1362, 1132, 5752);
				one_small_sphere(119, 174, 83, -114, 660, 6897);
				one_small_sphere(150, 60, 40, 894, 851, 6331);
				one_small_sphere(140, 96, 96, 296, 297, 8300);
				one_small_sphere(27, 111, 170, 263, 616, 6981);
				one_small_sphere(111, 48, 126, 550, 315, 8189);
				one_small_sphere(215, 56, 205, 890, 779, 6519);
				one_small_sphere(195, 9, 129, 414, 510, 7337);
				one_small_sphere(252, 190, 0, -299, 845, 6414);
				one_small_sphere(192, 4, 136, 748, 391, 7829);
				one_small_sphere(203, 40, 189, 170, 346, 8292);
				one_small_sphere(21, 146, 40, 489, 288, 8424);
				one_small_sphere(97, 35, 61, 616, 876, 6297);

			}

			// #2
			if (0)
			{
				one_small_sphere(37, 251, 193, 964, 1233, 5483);
				one_small_sphere(235, 36, 214, -227, 1008, 6000);
				one_small_sphere(97, 79, 236, 849, 411, 7753);
				one_small_sphere(133, 143, 39, 635, 322, 8154);
				one_small_sphere(203, 143, 186, -361, 911, 6260);
				one_small_sphere(21, 159, 156, -513, 564, 7334);
				one_small_sphere(162, 182, 209, 237, 1265, 5405);
				one_small_sphere(25, 215, 162, 1082, 962, 6083);
				one_small_sphere(37, 135, 171, -136, 511, 7409);
				one_small_sphere(46, 252, 176, 444, 614, 6979);
				one_small_sphere(208, 43, 58, -506, 695, 6890);
				one_small_sphere(97, 113, 29, 887, 746, 6610);
				one_small_sphere(218, 129, 61, 1271, 286, 8546);
				one_small_sphere(58, 9, 17, 1335, 474, 7612);
				one_small_sphere(162, 232, 72, 146, 488, 7441);
				one_small_sphere(105, 82, 35, 1331, 522, 7423);
				one_small_sphere(90, 7, 140, -155, 804, 6491);
				one_small_sphere(110, 11, 70, 1175, 292, 8460);
				one_small_sphere(96, 130, 166, 1348, 1178, 5651);
				one_small_sphere(60, 243, 119, 87, 273, 8493);
				one_small_sphere(225, 207, 116, 44, 255, 8631);
				one_small_sphere(29, 85, 209, 563, 401, 7770);
				one_small_sphere(134, 26, 126, 488, 422, 7679);
				one_small_sphere(15, 79, 174, 997, 887, 6252);
				one_small_sphere(252, 225, 220, -391, 467, 7663);
				one_small_sphere(248, 180, 117, 238, 563, 7163);
				one_small_sphere(182, 215, 127, 1313, 797, 6543);
				one_small_sphere(70, 175, 175, -1, 675, 6830);
				one_small_sphere(48, 150, 59, 954, 254, 8635);
				one_small_sphere(156, 193, 48, 1243, 419, 7817);
				one_small_sphere(212, 82, 192, -364, 467, 7651);
				one_small_sphere(138, 142, 0, 485, 484, 7431);
				one_small_sphere(89, 238, 165, 1160, 268, 8615);
				one_small_sphere(161, 229, 135, 167, 1026, 5908);
				one_small_sphere(110, 212, 138, 1219, 261, 8701);
				one_small_sphere(161, 199, 170, 22, 1030, 5912);
				one_small_sphere(16, 182, 40, 1315, 1105, 5801);
				one_small_sphere(163, 7, 231, -505, 1139, 5767);
				one_small_sphere(220, 74, 43, 279, 248, 8623);
				one_small_sphere(189, 224, 143, 608, 313, 8205);
				one_small_sphere(126, 89, 168, 340, 423, 7682);
				one_small_sphere(234, 42, 175, 1229, 712, 6765);
				one_small_sphere(62, 172, 143, 555, 1267, 5393);
				one_small_sphere(244, 182, 244, -183, 581, 7172);
				one_small_sphere(128, 110, 19, -107, 781, 6546);
				one_small_sphere(190, 6, 198, 167, 576, 7125);
				one_small_sphere(164, 62, 46, 889, 541, 7255);
				one_small_sphere(159, 188, 182, 584, 264, 8496);
				one_small_sphere(134, 101, 40, 234, 928, 6129);
				one_small_sphere(126, 217, 200, -502, 1225, 5588);
				one_small_sphere(49, 107, 235, 1046, 307, 8322);
				one_small_sphere(196, 50, 205, 1299, 573, 7231);
				one_small_sphere(3, 196, 149, -164, 385, 7946);
				one_small_sphere(3, 118, 151, -401, 316, 8426);
				one_small_sphere(104, 34, 175, -395, 707, 6821);
				one_small_sphere(200, 107, 168, 689, 1092, 5753);
				one_small_sphere(232, 253, 180, -60, 301, 8362);
				one_small_sphere(250, 214, 115, 359, 864, 6279);
				one_small_sphere(115, 127, 177, 816, 440, 7626);
				one_small_sphere(165, 29, 129, 468, 476, 7465);
				one_small_sphere(119, 55, 36, -110, 1385, 5210);
				one_small_sphere(183, 175, 95, -464, 463, 7710);
				one_small_sphere(174, 19, 184, 1059, 437, 7685);
				one_small_sphere(193, 63, 172, 380, 419, 7696);
				one_small_sphere(224, 17, 138, 17, 947, 6104);
				one_small_sphere(15, 216, 117, -388, 653, 6985);
				one_small_sphere(165, 35, 226, -396, 328, 8345);
				one_small_sphere(57, 3, 15, 887, 522, 7320);
				one_small_sphere(8, 200, 57, 1281, 933, 6188);
				one_small_sphere(97, 120, 236, 449, 487, 7423);
				one_small_sphere(100, 170, 43, 390, 622, 6954);
				one_small_sphere(176, 220, 179, 552, 360, 7958);
				one_small_sphere(223, 64, 8, 859, 337, 8105);
				one_small_sphere(95, 201, 246, 454, 485, 7428);
				one_small_sphere(196, 137, 179, 601, 239, 8674);
				one_small_sphere(126, 173, 44, 929, 765, 6563);
				one_small_sphere(166, 81, 141, 904, 400, 7810);
				one_small_sphere(15, 113, 185, 432, 959, 6047);
				one_small_sphere(144, 144, 24, 190, 940, 6103);
				one_small_sphere(75, 17, 190, -68, 1168, 5627);
				one_small_sphere(18, 95, 167, -270, 319, 8335);
				one_small_sphere(69, 179, 131, 1315, 1296, 5409);
				one_small_sphere(42, 227, 217, -344, 1310, 5388);
				one_small_sphere(236, 193, 135, 697, 1308, 5319);
				one_small_sphere(25, 170, 221, 255, 358, 7984);
				one_small_sphere(127, 132, 145, 950, 600, 7060);
				one_small_sphere(20, 106, 146, 343, 1201, 5521);
				one_small_sphere(109, 158, 213, -338, 1265, 5411);
			}

			// #1
			if (1)
			{
				one_small_sphere(177, 98, 121, 1222, 450, 7675);
				one_small_sphere(147, 38, 170, 1218, 522, 7388);
				one_small_sphere(45, 250, 114, 439, 1300, 5330);
				one_small_sphere(200, 233, 172, 1051, 550, 7248);
				one_small_sphere(30, 63, 32, 1141, 295, 8426);
				one_small_sphere(56, 147, 171, 137, 1162, 5617);
				one_small_sphere(243, 112, 214, 614, 342, 8049);
				one_small_sphere(13, 75, 201, 1063, 1144, 5677);
				one_small_sphere(97, 134, 140, 1162, 656, 6917);
				one_small_sphere(109, 211, 143, -70, 523, 7349);
				one_small_sphere(148, 46, 40, 36, 509, 7379);
				one_small_sphere(207, 178, 214, 1045, 1204, 5550);
				one_small_sphere(91, 30, 217, -40, 905, 6213);
				one_small_sphere(166, 203, 111, 1109, 272, 8568);
				one_small_sphere(62, 222, 63, 542, 404, 7757);
				one_small_sphere(21, 248, 159, 1316, 894, 6291);
				one_small_sphere(79, 236, 52, -463, 1200, 5631);
				one_small_sphere(49, 236, 183, 811, 497, 7401);
				one_small_sphere(226, 186, 97, 1040, 465, 7566);
				one_small_sphere(128, 19, 231, 471, 259, 8527);
				one_small_sphere(47, 181, 23, 118, 323, 8184);
				one_small_sphere(39, 197, 56, -248, 464, 7625);
				one_small_sphere(64, 36, 233, 1140, 815, 6459);
				one_small_sphere(236, 98, 191, 493, 648, 6872);
				one_small_sphere(120, 57, 105, 968, 1202, 5546);
				one_small_sphere(109, 126, 25, 1360, 1047, 5938);
				one_small_sphere(2, 180, 31, 1185, 1050, 5898);
				one_small_sphere(229, 63, 115, 1350, 541, 7360);
				one_small_sphere(212, 75, 49, 374, 381, 7862);
				one_small_sphere(189, 104, 237, 930, 426, 7705);
				one_small_sphere(121, 112, 251, 154, 884, 6243);
				one_small_sphere(149, 87, 3, 650, 227, 8771);
				one_small_sphere(139, 25, 36, 474, 269, 8460);
				one_small_sphere(229, 51, 144, 387, 517, 7313);
				one_small_sphere(106, 29, 239, 485, 456, 7540);
				one_small_sphere(2, 202, 223, 856, 1233, 5473);
				one_small_sphere(254, 67, 210, 1065, 752, 6617);
				one_small_sphere(47, 12, 17, -516, 1166, 5713);
				one_small_sphere(108, 101, 83, 322, 426, 7668);
				one_small_sphere(38, 165, 221, 1005, 1102, 5758);
				one_small_sphere(204, 160, 192, 1266, 385, 7984);
				one_small_sphere(177, 32, 70, 1334, 273, 8680);
				one_small_sphere(0, 65, 36, 756, 1147, 5641);
				one_small_sphere(195, 38, 12, 1058, 320, 8248);
				one_small_sphere(187, 169, 28, -381, 667, 6940);
				one_small_sphere(247, 80, 216, 193, 257, 8576);
				one_small_sphere(50, 222, 59, 117, 257, 8589);
				one_small_sphere(116, 224, 240, -234, 1135, 5720);
				one_small_sphere(141, 126, 236, -193, 359, 8084);
				one_small_sphere(38, 30, 101, -117, 1028, 5936);
				one_small_sphere(122, 236, 17, 1309, 292, 8525);
				one_small_sphere(221, 54, 206, 1280, 259, 8748);
				one_small_sphere(117, 131, 225, 1334, 956, 6143);
				one_small_sphere(168, 126, 128, 438, 440, 7606);
				one_small_sphere(52, 17, 64, -330, 1116, 5780);
				one_small_sphere(141, 185, 113, 432, 1193, 5540);
				one_small_sphere(45, 244, 42, 894, 265, 8538);
				one_small_sphere(140, 60, 37, 1197, 406, 7861);
				one_small_sphere(218, 109, 102, -498, 664, 6984);
				one_small_sphere(94, 235, 187, 1336, 1310, 5386);
				one_small_sphere(253, 123, 112, 1217, 328, 8263);
				one_small_sphere(3, 1, 130, 787, 657, 6858);
				one_small_sphere(230, 239, 8, -272, 1367, 5266);
				one_small_sphere(245, 105, 84, 414, 275, 8423);
				one_small_sphere(48, 41, 224, 507, 238, 8680);
				one_small_sphere(181, 56, 153, -8, 865, 6309);
				one_small_sphere(119, 227, 146, 1283, 709, 6787);
				one_small_sphere(162, 124, 194, 208, 1198, 5539);
				one_small_sphere(173, 83, 230, 562, 237, 8684);
				one_small_sphere(111, 226, 20, 144, 466, 7526);
				one_small_sphere(160, 234, 60, 576, 305, 8245);
				one_small_sphere(72, 40, 213, 1291, 1071, 5869);
				one_small_sphere(225, 67, 248, 956, 315, 8248);
				one_small_sphere(198, 9, 96, 801, 1097, 5749);
				one_small_sphere(19, 232, 55, 39, 1191, 5567);
				one_small_sphere(146, 157, 153, 279, 949, 6076);
				one_small_sphere(10, 173, 31, 313, 720, 6665);
				one_small_sphere(186, 231, 110, 237, 501, 7382);
				one_small_sphere(113, 247, 16, 1057, 627, 6989);
				one_small_sphere(23, 180, 200, 391, 808, 6421);
				one_small_sphere(140, 95, 20, 517, 1004, 5944);
				one_small_sphere(115, 3, 19, 531, 1346, 5244);
				one_small_sphere(3, 125, 45, 91, 245, 8687);
				one_small_sphere(175, 86, 113, 528, 227, 8765);
				one_small_sphere(10, 162, 39, 223, 254, 8591);
				one_small_sphere(173, 179, 116, -19, 732, 6668);
				one_small_sphere(156, 108, 65, 70, 587, 7103);
				one_small_sphere(163, 111, 59, -293, 1152, 5696);
				one_small_sphere(188, 195, 198, 634, 391, 7819);
				one_small_sphere(235, 248, 14, 732, 867, 6277);
				one_small_sphere(109, 79, 26, 813, 1293, 5353);
				one_small_sphere(116, 96, 137, 799, 249, 8626);
				one_small_sphere(219, 40, 72, 355, 1043, 5858);
				one_small_sphere(238, 119, 28, 1003, 392, 7871);
				one_small_sphere(145, 41, 187, -69, 1367, 5238);
				one_small_sphere(226, 0, 105, 968, 709, 6726);
				one_small_sphere(19, 120, 107, 860, 513, 7351);
				one_small_sphere(174, 132, 138, 766, 355, 8001);
				one_small_sphere(138, 199, 20, 1310, 1385, 5240);
				one_small_sphere(129, 25, 131, 846, 1343, 5261);
				one_small_sphere(196, 69, 159, -99, 387, 7915);
				one_small_sphere(178, 185, 48, 249, 1343, 5255);
				one_small_sphere(130, 146, 195, -76, 1055, 5869);
				one_small_sphere(91, 16, 253, 1349, 1134, 5745);
				one_small_sphere(43, 160, 132, 537, 1341, 5253);
				one_small_sphere(30, 203, 254, 206, 1347, 5251);
				one_small_sphere(131, 114, 212, 824, 432, 7661);
				one_small_sphere(30, 2, 142, -475, 386, 8067);
				one_small_sphere(13, 157, 70, 286, 1148, 5636);
				one_small_sphere(190, 47, 93, 1307, 813, 6499);
				one_small_sphere(0, 224, 53, 280, 762, 6550);
				one_small_sphere(192, 147, 37, 528, 259, 8528);
				one_small_sphere(145, 171, 171, -5, 269, 8548);
				one_small_sphere(240, 93, 93, -296, 431, 7779);
				one_small_sphere(214, 175, 2, 1085, 263, 8617);
				one_small_sphere(134, 78, 103, 1010, 1302, 5354);
				one_small_sphere(123, 67, 141, -263, 1135, 5727);
				one_small_sphere(188, 24, 15, -482, 1016, 6035);
				one_small_sphere(173, 176, 204, 67, 1203, 5541);
				one_small_sphere(114, 161, 114, 825, 499, 7398);
				one_small_sphere(196, 207, 15, 905, 972, 6038);
				one_small_sphere(101, 159, 144, 700, 292, 8331);
				one_small_sphere(210, 49, 253, 231, 1220, 5493);
				one_small_sphere(132, 7, 25, 866, 1242, 5457);
				one_small_sphere(228, 93, 64, 1305, 537, 7360);
				one_small_sphere(220, 55, 33, 134, 787, 6494);
				one_small_sphere(126, 249, 130, -83, 826, 6422);
				one_small_sphere(191, 88, 139, -380, 285, 8616);
				one_small_sphere(75, 222, 136, 522, 1267, 5393);
				one_small_sphere(209, 206, 218, 663, 658, 6845);
				one_small_sphere(157, 84, 206, -450, 1280, 5467);
				one_small_sphere(232, 187, 161, -14, 461, 7576);
				one_small_sphere(68, 36, 131, 448, 899, 6190);
				one_small_sphere(201, 216, 220, 1103, 1331, 5309);
				one_small_sphere(137, 63, 125, 37, 902, 6211);
				one_small_sphere(252, 221, 53, 74, 1096, 5763);
				one_small_sphere(89, 220, 56, -269, 720, 6751);
				one_small_sphere(252, 4, 121, 549, 545, 7211);
				one_small_sphere(205, 64, 16, 365, 983, 5994);
				one_small_sphere(64, 191, 84, -232, 710, 6771);
				one_small_sphere(234, 161, 185, 726, 1219, 5493);
				one_small_sphere(104, 100, 198, -515, 442, 7822);
				one_small_sphere(230, 30, 138, -325, 621, 7071);
				one_small_sphere(108, 223, 29, 787, 411, 7747);
				one_small_sphere(4, 118, 182, 1194, 893, 6270);
				one_small_sphere(199, 84, 152, 1100, 958, 6094);
				one_small_sphere(145, 216, 250, -35, 270, 8550);
				one_small_sphere(101, 124, 27, 503, 647, 6876);
				one_small_sphere(186, 200, 123, -368, 502, 7511);
				one_small_sphere(108, 130, 62, 520, 357, 7971);
			}

			
			//one_small_sphere()

			// Wiele ma�ych
			scene->add_sphere(
				d3(center.x, G::HEIGHT + u(-400), center.z + rad - u(3000)), small_rad,
				u(0), u(0), Surface_type::diffuse,
				RGB(255, 140, 0)
			);
		}
	}

	setup_DEFAULT_details;

	setup_END;
}

// 30 mirrorów / 155 wszystkich
void Setuper::setup_scene_6(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	unit rad = u(10000);
	unit small_rad = u(50);
	d3 ground(center.x, G::HEIGHT + u(0.92) * rad, center.z + rad);

	// 1 o�wietlaj�ce wszystkie
	{
		// G��WNE
		scene->add_light(
			d3(center.x, G::HEIGHT - u(1500), center.z + u(6000))
			, RGB(255, 255, 255));
	}

	// POD�O�E
	{
		scene->add_sphere(
			ground, rad,
			u(0), u(0),
			Surface_type::diffuse,
			//Surface_type::mirror,
			//RGB(255, 140, 0)
			RGB(109, 114, 122)
		);
	}

	// KILKA MA�YCH
	{
		unit y = G::HEIGHT + u(-1000) + u(450);

		// 3 na g�rze
		{
			//// Du�a
			scene->add_sphere(
				d3(center.x - u(350), y, center.z + rad - u(3000)), u(200),
				u(0), u(0),
				Surface_type::diffuse,
				//Surface_type::mirror,
				RGB(255, 0, 123)
			);

			//// �rednia
			scene->add_sphere(
				d3(center.x, y + u(50), center.z + rad - u(3000)), u(150),
				u(0), u(0), Surface_type::diffuse,
				RGB(0, 255, 123)
			);

			//// Ma�a
			scene->add_sphere(
				d3(center.x + u(260), y + (88), center.z + rad - u(3000)), u(112),
				u(0), u(0), Surface_type::diffuse,
				RGB(123, 0, 255)
			);
		}

		// wcze�niej wylosowane teraz na sta�e
		{
			#define one_small_sphere_mirrorr(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::mirror, \
				RGB(r, g, b) \
				); \
			}

			#define one_small_sphere_diffuse(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::diffuse, \
				RGB(r, g, b) \
				); \
			}
			
			{
				one_small_sphere_mirrorr(177, 98, 121, 1222, 450, 7675);
				one_small_sphere_mirrorr(147, 38, 170, 1218, 522, 7388);
				one_small_sphere_mirrorr(45, 250, 114, 439, 1300, 5330);
				one_small_sphere_mirrorr(200, 233, 172, 1051, 550, 7248);
				one_small_sphere_mirrorr(30, 63, 32, 1141, 295, 8426);
				one_small_sphere_mirrorr(56, 147, 171, 137, 1162, 5617);
				one_small_sphere_mirrorr(243, 112, 214, 614, 342, 8049);
				one_small_sphere_mirrorr(13, 75, 201, 1063, 1144, 5677);
				one_small_sphere_mirrorr(97, 134, 140, 1162, 656, 6917);
				one_small_sphere_mirrorr(109, 211, 143, -70, 523, 7349);
				one_small_sphere_mirrorr(148, 46, 40, 36, 509, 7379);
				one_small_sphere_mirrorr(207, 178, 214, 1045, 1204, 5550);
				one_small_sphere_mirrorr(91, 30, 217, -40, 905, 6213);
				one_small_sphere_mirrorr(166, 203, 111, 1109, 272, 8568);
				one_small_sphere_mirrorr(62, 222, 63, 542, 404, 7757);
				one_small_sphere_mirrorr(21, 248, 159, 1316, 894, 6291);
				one_small_sphere_mirrorr(79, 236, 52, -463, 1200, 5631);
				one_small_sphere_mirrorr(49, 236, 183, 811, 497, 7401);
				one_small_sphere_mirrorr(226, 186, 97, 1040, 465, 7566);
				one_small_sphere_mirrorr(128, 19, 231, 471, 259, 8527);
				one_small_sphere_mirrorr(47, 181, 23, 118, 323, 8184);
				one_small_sphere_mirrorr(39, 197, 56, -248, 464, 7625);
				one_small_sphere_mirrorr(64, 36, 233, 1140, 815, 6459);
				one_small_sphere_mirrorr(236, 98, 191, 493, 648, 6872);
				one_small_sphere_mirrorr(120, 57, 105, 968, 1202, 5546);
				one_small_sphere_mirrorr(109, 126, 25, 1360, 1047, 5938);
				one_small_sphere_mirrorr(2, 180, 31, 1185, 1050, 5898);
				one_small_sphere_mirrorr(229, 63, 115, 1350, 541, 7360);
				one_small_sphere_mirrorr(212, 75, 49, 374, 381, 7862);
				one_small_sphere_mirrorr(189, 104, 237, 930, 426, 7705);
				one_small_sphere_mirrorr(121, 112, 251, 154, 884, 6243);
				one_small_sphere_mirrorr(149, 87, 3, 650, 227, 8771);
				one_small_sphere_mirrorr(139, 25, 36, 474, 269, 8460);
				one_small_sphere_mirrorr(229, 51, 144, 387, 517, 7313);
				one_small_sphere_mirrorr(106, 29, 239, 485, 456, 7540);
				one_small_sphere_mirrorr(2, 202, 223, 856, 1233, 5473);
				one_small_sphere_mirrorr(254, 67, 210, 1065, 752, 6617);
				one_small_sphere_mirrorr(47, 12, 17, -516, 1166, 5713);
				one_small_sphere_mirrorr(108, 101, 83, 322, 426, 7668);
				one_small_sphere_diffuse(38, 165, 221, 1005, 1102, 5758);
				one_small_sphere_diffuse(204, 160, 192, 1266, 385, 7984);
				one_small_sphere_diffuse(177, 32, 70, 1334, 273, 8680);
				one_small_sphere_diffuse(0, 65, 36, 756, 1147, 5641);
				one_small_sphere_diffuse(195, 38, 12, 1058, 320, 8248);
				one_small_sphere_diffuse(187, 169, 28, -381, 667, 6940);
				one_small_sphere_diffuse(247, 80, 216, 193, 257, 8576);
				one_small_sphere_diffuse(50, 222, 59, 117, 257, 8589);
				one_small_sphere_diffuse(116, 224, 240, -234, 1135, 5720);
				one_small_sphere_diffuse(141, 126, 236, -193, 359, 8084);
				one_small_sphere_diffuse(38, 30, 101, -117, 1028, 5936);
				one_small_sphere_diffuse(122, 236, 17, 1309, 292, 8525);
				one_small_sphere_diffuse(221, 54, 206, 1280, 259, 8748);
				one_small_sphere_diffuse(117, 131, 225, 1334, 956, 6143);
				one_small_sphere_diffuse(168, 126, 128, 438, 440, 7606);
				one_small_sphere_diffuse(52, 17, 64, -330, 1116, 5780);
				one_small_sphere_diffuse(141, 185, 113, 432, 1193, 5540);
				one_small_sphere_diffuse(45, 244, 42, 894, 265, 8538);
				one_small_sphere_diffuse(140, 60, 37, 1197, 406, 7861);
				one_small_sphere_diffuse(218, 109, 102, -498, 664, 6984);
				one_small_sphere_diffuse(94, 235, 187, 1336, 1310, 5386);
				one_small_sphere_diffuse(253, 123, 112, 1217, 328, 8263);
				one_small_sphere_diffuse(3, 1, 130, 787, 657, 6858);
				one_small_sphere_diffuse(230, 239, 8, -272, 1367, 5266);
				one_small_sphere_diffuse(245, 105, 84, 414, 275, 8423);
				one_small_sphere_diffuse(48, 41, 224, 507, 238, 8680);
				one_small_sphere_diffuse(181, 56, 153, -8, 865, 6309);
				one_small_sphere_diffuse(119, 227, 146, 1283, 709, 6787);
				one_small_sphere_diffuse(162, 124, 194, 208, 1198, 5539);
				one_small_sphere_diffuse(173, 83, 230, 562, 237, 8684);
				one_small_sphere_diffuse(111, 226, 20, 144, 466, 7526);
				one_small_sphere_diffuse(160, 234, 60, 576, 305, 8245);
				one_small_sphere_diffuse(72, 40, 213, 1291, 1071, 5869);
				one_small_sphere_diffuse(225, 67, 248, 956, 315, 8248);
				one_small_sphere_diffuse(198, 9, 96, 801, 1097, 5749);
				one_small_sphere_diffuse(19, 232, 55, 39, 1191, 5567);
				one_small_sphere_diffuse(146, 157, 153, 279, 949, 6076);
				one_small_sphere_diffuse(10, 173, 31, 313, 720, 6665);
				one_small_sphere_diffuse(186, 231, 110, 237, 501, 7382);
				one_small_sphere_diffuse(113, 247, 16, 1057, 627, 6989);
				one_small_sphere_diffuse(23, 180, 200, 391, 808, 6421);
				one_small_sphere_diffuse(140, 95, 20, 517, 1004, 5944);
				one_small_sphere_diffuse(115, 3, 19, 531, 1346, 5244);
				one_small_sphere_diffuse(3, 125, 45, 91, 245, 8687);
				one_small_sphere_diffuse(175, 86, 113, 528, 227, 8765);
				one_small_sphere_diffuse(10, 162, 39, 223, 254, 8591);
				one_small_sphere_diffuse(173, 179, 116, -19, 732, 6668);
				one_small_sphere_diffuse(156, 108, 65, 70, 587, 7103);
				one_small_sphere_diffuse(163, 111, 59, -293, 1152, 5696);
				one_small_sphere_diffuse(188, 195, 198, 634, 391, 7819);
				one_small_sphere_diffuse(235, 248, 14, 732, 867, 6277);
				one_small_sphere_diffuse(109, 79, 26, 813, 1293, 5353);
				one_small_sphere_diffuse(116, 96, 137, 799, 249, 8626);
				one_small_sphere_diffuse(219, 40, 72, 355, 1043, 5858);
				one_small_sphere_diffuse(238, 119, 28, 1003, 392, 7871);
				one_small_sphere_diffuse(145, 41, 187, -69, 1367, 5238);
				one_small_sphere_diffuse(226, 0, 105, 968, 709, 6726);
				one_small_sphere_diffuse(19, 120, 107, 860, 513, 7351);
				one_small_sphere_diffuse(174, 132, 138, 766, 355, 8001);
				one_small_sphere_diffuse(138, 199, 20, 1310, 1385, 5240);
				one_small_sphere_diffuse(129, 25, 131, 846, 1343, 5261);
				one_small_sphere_diffuse(196, 69, 159, -99, 387, 7915);
				one_small_sphere_diffuse(178, 185, 48, 249, 1343, 5255);
				one_small_sphere_diffuse(130, 146, 195, -76, 1055, 5869);
				one_small_sphere_diffuse(91, 16, 253, 1349, 1134, 5745);
				one_small_sphere_diffuse(43, 160, 132, 537, 1341, 5253);
				one_small_sphere_diffuse(30, 203, 254, 206, 1347, 5251);
				one_small_sphere_diffuse(131, 114, 212, 824, 432, 7661);
				one_small_sphere_diffuse(30, 2, 142, -475, 386, 8067);
				one_small_sphere_diffuse(13, 157, 70, 286, 1148, 5636);
				one_small_sphere_diffuse(190, 47, 93, 1307, 813, 6499);
				one_small_sphere_diffuse(0, 224, 53, 280, 762, 6550);
				one_small_sphere_diffuse(192, 147, 37, 528, 259, 8528);
				one_small_sphere_diffuse(145, 171, 171, -5, 269, 8548);
				one_small_sphere_diffuse(240, 93, 93, -296, 431, 7779);
				one_small_sphere_diffuse(214, 175, 2, 1085, 263, 8617);
				one_small_sphere_diffuse(134, 78, 103, 1010, 1302, 5354);
				one_small_sphere_diffuse(123, 67, 141, -263, 1135, 5727);
				one_small_sphere_diffuse(188, 24, 15, -482, 1016, 6035);
				one_small_sphere_diffuse(173, 176, 204, 67, 1203, 5541);
				one_small_sphere_diffuse(114, 161, 114, 825, 499, 7398);
				one_small_sphere_diffuse(196, 207, 15, 905, 972, 6038);
				one_small_sphere_diffuse(101, 159, 144, 700, 292, 8331);
				one_small_sphere_diffuse(210, 49, 253, 231, 1220, 5493);
				one_small_sphere_diffuse(132, 7, 25, 866, 1242, 5457);
				one_small_sphere_diffuse(228, 93, 64, 1305, 537, 7360);
				one_small_sphere_diffuse(220, 55, 33, 134, 787, 6494);
				one_small_sphere_diffuse(126, 249, 130, -83, 826, 6422);
				one_small_sphere_diffuse(191, 88, 139, -380, 285, 8616);
				one_small_sphere_diffuse(75, 222, 136, 522, 1267, 5393);
				one_small_sphere_diffuse(209, 206, 218, 663, 658, 6845);
				one_small_sphere_diffuse(157, 84, 206, -450, 1280, 5467);
				one_small_sphere_diffuse(232, 187, 161, -14, 461, 7576);
				one_small_sphere_diffuse(68, 36, 131, 448, 899, 6190);
				one_small_sphere_diffuse(201, 216, 220, 1103, 1331, 5309);
				one_small_sphere_diffuse(137, 63, 125, 37, 902, 6211);
				one_small_sphere_diffuse(252, 221, 53, 74, 1096, 5763);
				one_small_sphere_diffuse(89, 220, 56, -269, 720, 6751);
				one_small_sphere_diffuse(252, 4, 121, 549, 545, 7211);
				one_small_sphere_diffuse(205, 64, 16, 365, 983, 5994);
				one_small_sphere_diffuse(64, 191, 84, -232, 710, 6771);
				one_small_sphere_diffuse(234, 161, 185, 726, 1219, 5493);
				one_small_sphere_diffuse(104, 100, 198, -515, 442, 7822);
				one_small_sphere_diffuse(230, 30, 138, -325, 621, 7071);
				one_small_sphere_diffuse(108, 223, 29, 787, 411, 7747);
				one_small_sphere_diffuse(4, 118, 182, 1194, 893, 6270);
				one_small_sphere_diffuse(199, 84, 152, 1100, 958, 6094);
				one_small_sphere_diffuse(145, 216, 250, -35, 270, 8550);
				one_small_sphere_diffuse(101, 124, 27, 503, 647, 6876);
				one_small_sphere_diffuse(186, 200, 123, -368, 502, 7511);
				one_small_sphere_diffuse(108, 130, 62, 520, 357, 7971);
			}

			// Wiele ma�ych
			scene->add_sphere(
				d3(center.x, G::HEIGHT + u(-400), center.z + rad - u(3000)), small_rad,
				u(0), u(0), Surface_type::diffuse,
				RGB(255, 140, 0)
			);
		}
	}

	setup_DEFAULT_details;

	setup_END;
}

// 75 mirrorów / 155 wszystkich
void Setuper::setup_scene_7(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	unit rad = u(10000);
	unit small_rad = u(50);
	d3 ground(center.x, G::HEIGHT + u(0.92) * rad, center.z + rad);

	// 1 o�wietlaj�ce wszystkie
	{
		// G��WNE
		scene->add_light(
			d3(center.x, G::HEIGHT - u(1500), center.z + u(6000))
			, RGB(255, 255, 255));
	}

	// POD�O�E
	{
		scene->add_sphere(
			ground, rad,
			u(0), u(0),
			Surface_type::diffuse,
			//Surface_type::mirror,
			//RGB(255, 140, 0)
			RGB(109, 114, 122)
		);
	}

	// KILKA MA�YCH
	{
		unit y = G::HEIGHT + u(-1000) + u(450);

		// 3 na g�rze
		{
			//// Du�a
			scene->add_sphere(
				d3(center.x - u(350), y, center.z + rad - u(3000)), u(200),
				u(0), u(0),
				Surface_type::diffuse,
				//Surface_type::mirror,
				RGB(255, 0, 123)
			);

			//// �rednia
			scene->add_sphere(
				d3(center.x, y + u(50), center.z + rad - u(3000)), u(150),
				u(0), u(0), Surface_type::diffuse,
				RGB(0, 255, 123)
			);

			//// Ma�a
			scene->add_sphere(
				d3(center.x + u(260), y + (88), center.z + rad - u(3000)), u(112),
				u(0), u(0), Surface_type::diffuse,
				RGB(123, 0, 255)
			);
		}

		// wcze�niej wylosowane teraz na sta�e
		{
			#define one_small_sphere_mirrorr(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::mirror, \
				RGB(r, g, b) \
				); \
			}

			#define one_small_sphere_diffuse(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::diffuse, \
				RGB(r, g, b) \
				); \
			}
			
			{
				one_small_sphere_mirrorr(177, 98, 121, 1222, 450, 7675);
				one_small_sphere_mirrorr(147, 38, 170, 1218, 522, 7388);
				one_small_sphere_mirrorr(45, 250, 114, 439, 1300, 5330);
				one_small_sphere_mirrorr(200, 233, 172, 1051, 550, 7248);
				one_small_sphere_mirrorr(30, 63, 32, 1141, 295, 8426);
				one_small_sphere_mirrorr(56, 147, 171, 137, 1162, 5617);
				one_small_sphere_mirrorr(243, 112, 214, 614, 342, 8049);
				one_small_sphere_mirrorr(13, 75, 201, 1063, 1144, 5677);
				one_small_sphere_mirrorr(97, 134, 140, 1162, 656, 6917);
				one_small_sphere_mirrorr(109, 211, 143, -70, 523, 7349);
				one_small_sphere_mirrorr(148, 46, 40, 36, 509, 7379);
				one_small_sphere_mirrorr(207, 178, 214, 1045, 1204, 5550);
				one_small_sphere_mirrorr(91, 30, 217, -40, 905, 6213);
				one_small_sphere_mirrorr(166, 203, 111, 1109, 272, 8568);
				one_small_sphere_mirrorr(62, 222, 63, 542, 404, 7757);
				one_small_sphere_mirrorr(21, 248, 159, 1316, 894, 6291);
				one_small_sphere_mirrorr(79, 236, 52, -463, 1200, 5631);
				one_small_sphere_mirrorr(49, 236, 183, 811, 497, 7401);
				one_small_sphere_mirrorr(226, 186, 97, 1040, 465, 7566);
				one_small_sphere_mirrorr(128, 19, 231, 471, 259, 8527);
				one_small_sphere_mirrorr(47, 181, 23, 118, 323, 8184);
				one_small_sphere_mirrorr(39, 197, 56, -248, 464, 7625);
				one_small_sphere_mirrorr(64, 36, 233, 1140, 815, 6459);
				one_small_sphere_mirrorr(236, 98, 191, 493, 648, 6872);
				one_small_sphere_mirrorr(120, 57, 105, 968, 1202, 5546);
				one_small_sphere_mirrorr(109, 126, 25, 1360, 1047, 5938);
				one_small_sphere_mirrorr(2, 180, 31, 1185, 1050, 5898);
				one_small_sphere_mirrorr(229, 63, 115, 1350, 541, 7360);
				one_small_sphere_mirrorr(212, 75, 49, 374, 381, 7862);
				one_small_sphere_mirrorr(189, 104, 237, 930, 426, 7705);
				one_small_sphere_mirrorr(121, 112, 251, 154, 884, 6243);
				one_small_sphere_mirrorr(149, 87, 3, 650, 227, 8771);
				one_small_sphere_mirrorr(139, 25, 36, 474, 269, 8460);
				one_small_sphere_mirrorr(229, 51, 144, 387, 517, 7313);
				one_small_sphere_mirrorr(106, 29, 239, 485, 456, 7540);
				one_small_sphere_mirrorr(2, 202, 223, 856, 1233, 5473);
				one_small_sphere_mirrorr(254, 67, 210, 1065, 752, 6617);
				one_small_sphere_mirrorr(47, 12, 17, -516, 1166, 5713);
				one_small_sphere_mirrorr(108, 101, 83, 322, 426, 7668);
				one_small_sphere_mirrorr(38, 165, 221, 1005, 1102, 5758);
				one_small_sphere_mirrorr(204, 160, 192, 1266, 385, 7984);
				one_small_sphere_mirrorr(177, 32, 70, 1334, 273, 8680);
				one_small_sphere_mirrorr(0, 65, 36, 756, 1147, 5641);
				one_small_sphere_mirrorr(195, 38, 12, 1058, 320, 8248);
				one_small_sphere_mirrorr(187, 169, 28, -381, 667, 6940);
				one_small_sphere_mirrorr(247, 80, 216, 193, 257, 8576);
				one_small_sphere_mirrorr(50, 222, 59, 117, 257, 8589);
				one_small_sphere_mirrorr(116, 224, 240, -234, 1135, 5720);
				one_small_sphere_mirrorr(141, 126, 236, -193, 359, 8084);
				one_small_sphere_mirrorr(38, 30, 101, -117, 1028, 5936);
				one_small_sphere_mirrorr(122, 236, 17, 1309, 292, 8525);
				one_small_sphere_mirrorr(221, 54, 206, 1280, 259, 8748);
				one_small_sphere_mirrorr(117, 131, 225, 1334, 956, 6143);
				one_small_sphere_mirrorr(168, 126, 128, 438, 440, 7606);
				one_small_sphere_mirrorr(52, 17, 64, -330, 1116, 5780);
				one_small_sphere_mirrorr(141, 185, 113, 432, 1193, 5540);
				one_small_sphere_mirrorr(45, 244, 42, 894, 265, 8538);
				one_small_sphere_mirrorr(140, 60, 37, 1197, 406, 7861);
				one_small_sphere_mirrorr(218, 109, 102, -498, 664, 6984);
				one_small_sphere_mirrorr(94, 235, 187, 1336, 1310, 5386);
				one_small_sphere_mirrorr(253, 123, 112, 1217, 328, 8263);
				one_small_sphere_mirrorr(3, 1, 130, 787, 657, 6858);
				one_small_sphere_mirrorr(230, 239, 8, -272, 1367, 5266);
				one_small_sphere_mirrorr(245, 105, 84, 414, 275, 8423);
				one_small_sphere_mirrorr(48, 41, 224, 507, 238, 8680);
				one_small_sphere_mirrorr(181, 56, 153, -8, 865, 6309);
				one_small_sphere_mirrorr(119, 227, 146, 1283, 709, 6787);
				one_small_sphere_mirrorr(162, 124, 194, 208, 1198, 5539);
				one_small_sphere_mirrorr(173, 83, 230, 562, 237, 8684);
				one_small_sphere_mirrorr(111, 226, 20, 144, 466, 7526);
				one_small_sphere_mirrorr(160, 234, 60, 576, 305, 8245);
				one_small_sphere_mirrorr(72, 40, 213, 1291, 1071, 5869);
				one_small_sphere_mirrorr(225, 67, 248, 956, 315, 8248);
				one_small_sphere_mirrorr(198, 9, 96, 801, 1097, 5749);
				one_small_sphere_mirrorr(19, 232, 55, 39, 1191, 5567);
				one_small_sphere_diffuse(146, 157, 153, 279, 949, 6076);
				one_small_sphere_diffuse(10, 173, 31, 313, 720, 6665);
				one_small_sphere_diffuse(186, 231, 110, 237, 501, 7382);
				one_small_sphere_diffuse(113, 247, 16, 1057, 627, 6989);
				one_small_sphere_diffuse(23, 180, 200, 391, 808, 6421);
				one_small_sphere_diffuse(140, 95, 20, 517, 1004, 5944);
				one_small_sphere_diffuse(115, 3, 19, 531, 1346, 5244);
				one_small_sphere_diffuse(3, 125, 45, 91, 245, 8687);
				one_small_sphere_diffuse(175, 86, 113, 528, 227, 8765);
				one_small_sphere_diffuse(10, 162, 39, 223, 254, 8591);
				one_small_sphere_diffuse(173, 179, 116, -19, 732, 6668);
				one_small_sphere_diffuse(156, 108, 65, 70, 587, 7103);
				one_small_sphere_diffuse(163, 111, 59, -293, 1152, 5696);
				one_small_sphere_diffuse(188, 195, 198, 634, 391, 7819);
				one_small_sphere_diffuse(235, 248, 14, 732, 867, 6277);
				one_small_sphere_diffuse(109, 79, 26, 813, 1293, 5353);
				one_small_sphere_diffuse(116, 96, 137, 799, 249, 8626);
				one_small_sphere_diffuse(219, 40, 72, 355, 1043, 5858);
				one_small_sphere_diffuse(238, 119, 28, 1003, 392, 7871);
				one_small_sphere_diffuse(145, 41, 187, -69, 1367, 5238);
				one_small_sphere_diffuse(226, 0, 105, 968, 709, 6726);
				one_small_sphere_diffuse(19, 120, 107, 860, 513, 7351);
				one_small_sphere_diffuse(174, 132, 138, 766, 355, 8001);
				one_small_sphere_diffuse(138, 199, 20, 1310, 1385, 5240);
				one_small_sphere_diffuse(129, 25, 131, 846, 1343, 5261);
				one_small_sphere_diffuse(196, 69, 159, -99, 387, 7915);
				one_small_sphere_diffuse(178, 185, 48, 249, 1343, 5255);
				one_small_sphere_diffuse(130, 146, 195, -76, 1055, 5869);
				one_small_sphere_diffuse(91, 16, 253, 1349, 1134, 5745);
				one_small_sphere_diffuse(43, 160, 132, 537, 1341, 5253);
				one_small_sphere_diffuse(30, 203, 254, 206, 1347, 5251);
				one_small_sphere_diffuse(131, 114, 212, 824, 432, 7661);
				one_small_sphere_diffuse(30, 2, 142, -475, 386, 8067);
				one_small_sphere_diffuse(13, 157, 70, 286, 1148, 5636);
				one_small_sphere_diffuse(190, 47, 93, 1307, 813, 6499);
				one_small_sphere_diffuse(0, 224, 53, 280, 762, 6550);
				one_small_sphere_diffuse(192, 147, 37, 528, 259, 8528);
				one_small_sphere_diffuse(145, 171, 171, -5, 269, 8548);
				one_small_sphere_diffuse(240, 93, 93, -296, 431, 7779);
				one_small_sphere_diffuse(214, 175, 2, 1085, 263, 8617);
				one_small_sphere_diffuse(134, 78, 103, 1010, 1302, 5354);
				one_small_sphere_diffuse(123, 67, 141, -263, 1135, 5727);
				one_small_sphere_diffuse(188, 24, 15, -482, 1016, 6035);
				one_small_sphere_diffuse(173, 176, 204, 67, 1203, 5541);
				one_small_sphere_diffuse(114, 161, 114, 825, 499, 7398);
				one_small_sphere_diffuse(196, 207, 15, 905, 972, 6038);
				one_small_sphere_diffuse(101, 159, 144, 700, 292, 8331);
				one_small_sphere_diffuse(210, 49, 253, 231, 1220, 5493);
				one_small_sphere_diffuse(132, 7, 25, 866, 1242, 5457);
				one_small_sphere_diffuse(228, 93, 64, 1305, 537, 7360);
				one_small_sphere_diffuse(220, 55, 33, 134, 787, 6494);
				one_small_sphere_diffuse(126, 249, 130, -83, 826, 6422);
				one_small_sphere_diffuse(191, 88, 139, -380, 285, 8616);
				one_small_sphere_diffuse(75, 222, 136, 522, 1267, 5393);
				one_small_sphere_diffuse(209, 206, 218, 663, 658, 6845);
				one_small_sphere_diffuse(157, 84, 206, -450, 1280, 5467);
				one_small_sphere_diffuse(232, 187, 161, -14, 461, 7576);
				one_small_sphere_diffuse(68, 36, 131, 448, 899, 6190);
				one_small_sphere_diffuse(201, 216, 220, 1103, 1331, 5309);
				one_small_sphere_diffuse(137, 63, 125, 37, 902, 6211);
				one_small_sphere_diffuse(252, 221, 53, 74, 1096, 5763);
				one_small_sphere_diffuse(89, 220, 56, -269, 720, 6751);
				one_small_sphere_diffuse(252, 4, 121, 549, 545, 7211);
				one_small_sphere_diffuse(205, 64, 16, 365, 983, 5994);
				one_small_sphere_diffuse(64, 191, 84, -232, 710, 6771);
				one_small_sphere_diffuse(234, 161, 185, 726, 1219, 5493);
				one_small_sphere_diffuse(104, 100, 198, -515, 442, 7822);
				one_small_sphere_diffuse(230, 30, 138, -325, 621, 7071);
				one_small_sphere_diffuse(108, 223, 29, 787, 411, 7747);
				one_small_sphere_diffuse(4, 118, 182, 1194, 893, 6270);
				one_small_sphere_diffuse(199, 84, 152, 1100, 958, 6094);
				one_small_sphere_diffuse(145, 216, 250, -35, 270, 8550);
				one_small_sphere_diffuse(101, 124, 27, 503, 647, 6876);
				one_small_sphere_diffuse(186, 200, 123, -368, 502, 7511);
				one_small_sphere_diffuse(108, 130, 62, 520, 357, 7971);
			}

			// Wiele ma�ych
			scene->add_sphere(
				d3(center.x, G::HEIGHT + u(-400), center.z + rad - u(3000)), small_rad,
				u(0), u(0), Surface_type::diffuse,
				RGB(255, 140, 0)
			);
		}
	}

	setup_DEFAULT_details;

	setup_END;
}

// 125 mirrorów / 155 wszystkich
void Setuper::setup_scene_8(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	unit rad = u(10000);
	unit small_rad = u(50);
	d3 ground(center.x, G::HEIGHT + u(0.92) * rad, center.z + rad);

	// 1 o�wietlaj�ce wszystkie
	{
		// G��WNE
		scene->add_light(
			d3(center.x, G::HEIGHT - u(1500), center.z + u(6000))
			, RGB(255, 255, 255));
	}

	// POD�O�E
	{
		scene->add_sphere(
			ground, rad,
			u(0), u(0),
			Surface_type::diffuse,
			//Surface_type::mirror,
			//RGB(255, 140, 0)
			RGB(109, 114, 122)
		);
	}

	// KILKA MA�YCH
	{
		unit y = G::HEIGHT + u(-1000) + u(450);

		// 3 na g�rze
		{
			//// Du�a
			scene->add_sphere(
				d3(center.x - u(350), y, center.z + rad - u(3000)), u(200),
				u(0), u(0),
				Surface_type::diffuse,
				//Surface_type::mirror,
				RGB(255, 0, 123)
			);

			//// �rednia
			scene->add_sphere(
				d3(center.x, y + u(50), center.z + rad - u(3000)), u(150),
				u(0), u(0), Surface_type::diffuse,
				RGB(0, 255, 123)
			);

			//// Ma�a
			scene->add_sphere(
				d3(center.x + u(260), y + (88), center.z + rad - u(3000)), u(112),
				u(0), u(0), Surface_type::diffuse,
				RGB(123, 0, 255)
			);
		}

		// wcze�niej wylosowane teraz na sta�e
		{
			#define one_small_sphere_mirrorr(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::mirror, \
				RGB(r, g, b) \
				); \
			}

			#define one_small_sphere_diffuse(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::diffuse, \
				RGB(r, g, b) \
				); \
			}
			
			{
				one_small_sphere_mirrorr(177, 98, 121, 1222, 450, 7675);
				one_small_sphere_mirrorr(147, 38, 170, 1218, 522, 7388);
				one_small_sphere_mirrorr(45, 250, 114, 439, 1300, 5330);
				one_small_sphere_mirrorr(200, 233, 172, 1051, 550, 7248);
				one_small_sphere_mirrorr(30, 63, 32, 1141, 295, 8426);
				one_small_sphere_mirrorr(56, 147, 171, 137, 1162, 5617);
				one_small_sphere_mirrorr(243, 112, 214, 614, 342, 8049);
				one_small_sphere_mirrorr(13, 75, 201, 1063, 1144, 5677);
				one_small_sphere_mirrorr(97, 134, 140, 1162, 656, 6917);
				one_small_sphere_mirrorr(109, 211, 143, -70, 523, 7349);
				one_small_sphere_mirrorr(148, 46, 40, 36, 509, 7379);
				one_small_sphere_mirrorr(207, 178, 214, 1045, 1204, 5550);
				one_small_sphere_mirrorr(91, 30, 217, -40, 905, 6213);
				one_small_sphere_mirrorr(166, 203, 111, 1109, 272, 8568);
				one_small_sphere_mirrorr(62, 222, 63, 542, 404, 7757);
				one_small_sphere_mirrorr(21, 248, 159, 1316, 894, 6291);
				one_small_sphere_mirrorr(79, 236, 52, -463, 1200, 5631);
				one_small_sphere_mirrorr(49, 236, 183, 811, 497, 7401);
				one_small_sphere_mirrorr(226, 186, 97, 1040, 465, 7566);
				one_small_sphere_mirrorr(128, 19, 231, 471, 259, 8527);
				one_small_sphere_mirrorr(47, 181, 23, 118, 323, 8184);
				one_small_sphere_mirrorr(39, 197, 56, -248, 464, 7625);
				one_small_sphere_mirrorr(64, 36, 233, 1140, 815, 6459);
				one_small_sphere_mirrorr(236, 98, 191, 493, 648, 6872);
				one_small_sphere_mirrorr(120, 57, 105, 968, 1202, 5546);
				one_small_sphere_mirrorr(109, 126, 25, 1360, 1047, 5938);
				one_small_sphere_mirrorr(2, 180, 31, 1185, 1050, 5898);
				one_small_sphere_mirrorr(229, 63, 115, 1350, 541, 7360);
				one_small_sphere_mirrorr(212, 75, 49, 374, 381, 7862);
				one_small_sphere_mirrorr(189, 104, 237, 930, 426, 7705);
				one_small_sphere_mirrorr(121, 112, 251, 154, 884, 6243);
				one_small_sphere_mirrorr(149, 87, 3, 650, 227, 8771);
				one_small_sphere_mirrorr(139, 25, 36, 474, 269, 8460);
				one_small_sphere_mirrorr(229, 51, 144, 387, 517, 7313);
				one_small_sphere_mirrorr(106, 29, 239, 485, 456, 7540);
				one_small_sphere_mirrorr(2, 202, 223, 856, 1233, 5473);
				one_small_sphere_mirrorr(254, 67, 210, 1065, 752, 6617);
				one_small_sphere_mirrorr(47, 12, 17, -516, 1166, 5713);
				one_small_sphere_mirrorr(108, 101, 83, 322, 426, 7668);
				one_small_sphere_mirrorr(38, 165, 221, 1005, 1102, 5758);
				one_small_sphere_mirrorr(204, 160, 192, 1266, 385, 7984);
				one_small_sphere_mirrorr(177, 32, 70, 1334, 273, 8680);
				one_small_sphere_mirrorr(0, 65, 36, 756, 1147, 5641);
				one_small_sphere_mirrorr(195, 38, 12, 1058, 320, 8248);
				one_small_sphere_mirrorr(187, 169, 28, -381, 667, 6940);
				one_small_sphere_mirrorr(247, 80, 216, 193, 257, 8576);
				one_small_sphere_mirrorr(50, 222, 59, 117, 257, 8589);
				one_small_sphere_mirrorr(116, 224, 240, -234, 1135, 5720);
				one_small_sphere_mirrorr(141, 126, 236, -193, 359, 8084);
				one_small_sphere_mirrorr(38, 30, 101, -117, 1028, 5936);
				one_small_sphere_mirrorr(122, 236, 17, 1309, 292, 8525);
				one_small_sphere_mirrorr(221, 54, 206, 1280, 259, 8748);
				one_small_sphere_mirrorr(117, 131, 225, 1334, 956, 6143);
				one_small_sphere_mirrorr(168, 126, 128, 438, 440, 7606);
				one_small_sphere_mirrorr(52, 17, 64, -330, 1116, 5780);
				one_small_sphere_mirrorr(141, 185, 113, 432, 1193, 5540);
				one_small_sphere_mirrorr(45, 244, 42, 894, 265, 8538);
				one_small_sphere_mirrorr(140, 60, 37, 1197, 406, 7861);
				one_small_sphere_mirrorr(218, 109, 102, -498, 664, 6984);
				one_small_sphere_mirrorr(94, 235, 187, 1336, 1310, 5386);
				one_small_sphere_mirrorr(253, 123, 112, 1217, 328, 8263);
				one_small_sphere_mirrorr(3, 1, 130, 787, 657, 6858);
				one_small_sphere_mirrorr(230, 239, 8, -272, 1367, 5266);
				one_small_sphere_mirrorr(245, 105, 84, 414, 275, 8423);
				one_small_sphere_mirrorr(48, 41, 224, 507, 238, 8680);
				one_small_sphere_mirrorr(181, 56, 153, -8, 865, 6309);
				one_small_sphere_mirrorr(119, 227, 146, 1283, 709, 6787);
				one_small_sphere_mirrorr(162, 124, 194, 208, 1198, 5539);
				one_small_sphere_mirrorr(173, 83, 230, 562, 237, 8684);
				one_small_sphere_mirrorr(111, 226, 20, 144, 466, 7526);
				one_small_sphere_mirrorr(160, 234, 60, 576, 305, 8245);
				one_small_sphere_mirrorr(72, 40, 213, 1291, 1071, 5869);
				one_small_sphere_mirrorr(225, 67, 248, 956, 315, 8248);
				one_small_sphere_mirrorr(198, 9, 96, 801, 1097, 5749);
				one_small_sphere_mirrorr(19, 232, 55, 39, 1191, 5567);
				one_small_sphere_mirrorr(146, 157, 153, 279, 949, 6076);
				one_small_sphere_mirrorr(10, 173, 31, 313, 720, 6665);
				one_small_sphere_mirrorr(186, 231, 110, 237, 501, 7382);
				one_small_sphere_mirrorr(113, 247, 16, 1057, 627, 6989);
				one_small_sphere_mirrorr(23, 180, 200, 391, 808, 6421);
				one_small_sphere_mirrorr(140, 95, 20, 517, 1004, 5944);
				one_small_sphere_mirrorr(115, 3, 19, 531, 1346, 5244);
				one_small_sphere_mirrorr(3, 125, 45, 91, 245, 8687);
				one_small_sphere_mirrorr(175, 86, 113, 528, 227, 8765);
				one_small_sphere_mirrorr(10, 162, 39, 223, 254, 8591);
				one_small_sphere_mirrorr(173, 179, 116, -19, 732, 6668);
				one_small_sphere_mirrorr(156, 108, 65, 70, 587, 7103);
				one_small_sphere_mirrorr(163, 111, 59, -293, 1152, 5696);
				one_small_sphere_mirrorr(188, 195, 198, 634, 391, 7819);
				one_small_sphere_mirrorr(235, 248, 14, 732, 867, 6277);
				one_small_sphere_mirrorr(109, 79, 26, 813, 1293, 5353);
				one_small_sphere_mirrorr(116, 96, 137, 799, 249, 8626);
				one_small_sphere_mirrorr(219, 40, 72, 355, 1043, 5858);
				one_small_sphere_mirrorr(238, 119, 28, 1003, 392, 7871);
				one_small_sphere_mirrorr(145, 41, 187, -69, 1367, 5238);
				one_small_sphere_mirrorr(226, 0, 105, 968, 709, 6726);
				one_small_sphere_mirrorr(19, 120, 107, 860, 513, 7351);
				one_small_sphere_mirrorr(174, 132, 138, 766, 355, 8001);
				one_small_sphere_mirrorr(138, 199, 20, 1310, 1385, 5240);
				one_small_sphere_mirrorr(129, 25, 131, 846, 1343, 5261);
				one_small_sphere_mirrorr(196, 69, 159, -99, 387, 7915);
				one_small_sphere_mirrorr(178, 185, 48, 249, 1343, 5255);
				one_small_sphere_mirrorr(130, 146, 195, -76, 1055, 5869);
				one_small_sphere_mirrorr(91, 16, 253, 1349, 1134, 5745);
				one_small_sphere_mirrorr(43, 160, 132, 537, 1341, 5253);
				one_small_sphere_mirrorr(30, 203, 254, 206, 1347, 5251);
				one_small_sphere_mirrorr(131, 114, 212, 824, 432, 7661);
				one_small_sphere_mirrorr(30, 2, 142, -475, 386, 8067);
				one_small_sphere_mirrorr(13, 157, 70, 286, 1148, 5636);
				one_small_sphere_mirrorr(190, 47, 93, 1307, 813, 6499);
				one_small_sphere_mirrorr(0, 224, 53, 280, 762, 6550);
				one_small_sphere_diffuse(192, 147, 37, 528, 259, 8528);
				one_small_sphere_diffuse(145, 171, 171, -5, 269, 8548);
				one_small_sphere_diffuse(240, 93, 93, -296, 431, 7779);
				one_small_sphere_diffuse(214, 175, 2, 1085, 263, 8617);
				one_small_sphere_diffuse(134, 78, 103, 1010, 1302, 5354);
				one_small_sphere_diffuse(123, 67, 141, -263, 1135, 5727);
				one_small_sphere_diffuse(188, 24, 15, -482, 1016, 6035);
				one_small_sphere_diffuse(173, 176, 204, 67, 1203, 5541);
				one_small_sphere_diffuse(114, 161, 114, 825, 499, 7398);
				one_small_sphere_diffuse(196, 207, 15, 905, 972, 6038);
				one_small_sphere_diffuse(101, 159, 144, 700, 292, 8331);
				one_small_sphere_diffuse(210, 49, 253, 231, 1220, 5493);
				one_small_sphere_diffuse(132, 7, 25, 866, 1242, 5457);
				one_small_sphere_diffuse(228, 93, 64, 1305, 537, 7360);
				one_small_sphere_diffuse(220, 55, 33, 134, 787, 6494);
				one_small_sphere_diffuse(126, 249, 130, -83, 826, 6422);
				one_small_sphere_diffuse(191, 88, 139, -380, 285, 8616);
				one_small_sphere_diffuse(75, 222, 136, 522, 1267, 5393);
				one_small_sphere_diffuse(209, 206, 218, 663, 658, 6845);
				one_small_sphere_diffuse(157, 84, 206, -450, 1280, 5467);
				one_small_sphere_diffuse(232, 187, 161, -14, 461, 7576);
				one_small_sphere_diffuse(68, 36, 131, 448, 899, 6190);
				one_small_sphere_diffuse(201, 216, 220, 1103, 1331, 5309);
				one_small_sphere_diffuse(137, 63, 125, 37, 902, 6211);
				one_small_sphere_diffuse(252, 221, 53, 74, 1096, 5763);
				one_small_sphere_diffuse(89, 220, 56, -269, 720, 6751);
				one_small_sphere_diffuse(252, 4, 121, 549, 545, 7211);
				one_small_sphere_diffuse(205, 64, 16, 365, 983, 5994);
				one_small_sphere_diffuse(64, 191, 84, -232, 710, 6771);
				one_small_sphere_diffuse(234, 161, 185, 726, 1219, 5493);
				one_small_sphere_diffuse(104, 100, 198, -515, 442, 7822);
				one_small_sphere_diffuse(230, 30, 138, -325, 621, 7071);
				one_small_sphere_diffuse(108, 223, 29, 787, 411, 7747);
				one_small_sphere_diffuse(4, 118, 182, 1194, 893, 6270);
				one_small_sphere_diffuse(199, 84, 152, 1100, 958, 6094);
				one_small_sphere_diffuse(145, 216, 250, -35, 270, 8550);
				one_small_sphere_diffuse(101, 124, 27, 503, 647, 6876);
				one_small_sphere_diffuse(186, 200, 123, -368, 502, 7511);
				one_small_sphere_diffuse(108, 130, 62, 520, 357, 7971);
			}

			// Wiele ma�ych
			scene->add_sphere(
				d3(center.x, G::HEIGHT + u(-400), center.z + rad - u(3000)), small_rad,
				u(0), u(0), Surface_type::diffuse,
				RGB(255, 140, 0)
			);
		}
	}

	setup_DEFAULT_details;

	setup_END;
}

// wszystkie to mirrory
void Setuper::setup_scene_9(Scene* scene, const string& name)
{
	setup_START(name);

	d3 center(u(G::WIDTH / 2), u(G::HEIGHT / 2), u(0));
	unit rad = u(10000);
	unit small_rad = u(50);
	d3 ground(center.x, G::HEIGHT + u(0.92) * rad, center.z + rad);

	// 1 o�wietlaj�ce wszystkie
	{
		// G��WNE
		scene->add_light(
			d3(center.x, G::HEIGHT - u(1500), center.z + u(6000))
			, RGB(255, 255, 255));
	}

	// POD�O�E
	{
		scene->add_sphere(
			ground, rad,
			u(0), u(0),
			Surface_type::diffuse,
			//Surface_type::mirror,
			//RGB(255, 140, 0)
			RGB(109, 114, 122)
		);
	}

	// KILKA MA�YCH
	{
		unit y = G::HEIGHT + u(-1000) + u(450);

		// 3 na g�rze
		{
			//// Du�a
			scene->add_sphere(
				d3(center.x - u(350), y, center.z + rad - u(3000)), u(200),
				u(0), u(0),
				//Surface_type::diffuse,
				Surface_type::mirror,
				RGB(255, 0, 123)
			);

			//// �rednia
			scene->add_sphere(
				d3(center.x, y + u(50), center.z + rad - u(3000)), u(150),
				u(0), u(0), Surface_type::mirror,
				RGB(0, 255, 123)
			);

			//// Ma�a
			scene->add_sphere(
				d3(center.x + u(260), y + (88), center.z + rad - u(3000)), u(112),
				u(0), u(0), Surface_type::mirror,
				RGB(123, 0, 255)
			);
		}

		// wcze�niej wylosowane teraz na sta�e
		{
			#define one_small_sphere_mirrorr(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::mirror, \
				RGB(r, g, b) \
				); \
			}

			#define one_small_sphere_diffuse(r, g, b, x, y, z) \
			{ \
				scene->add_sphere( \
				d3(u(x), u(y), u(z)), small_rad, \
				u(0), u(0), Surface_type::diffuse, \
				RGB(r, g, b) \
				); \
			}
			
			{
				one_small_sphere_mirrorr(177, 98, 121, 1222, 450, 7675);
				one_small_sphere_mirrorr(147, 38, 170, 1218, 522, 7388);
				one_small_sphere_mirrorr(45, 250, 114, 439, 1300, 5330);
				one_small_sphere_mirrorr(200, 233, 172, 1051, 550, 7248);
				one_small_sphere_mirrorr(30, 63, 32, 1141, 295, 8426);
				one_small_sphere_mirrorr(56, 147, 171, 137, 1162, 5617);
				one_small_sphere_mirrorr(243, 112, 214, 614, 342, 8049);
				one_small_sphere_mirrorr(13, 75, 201, 1063, 1144, 5677);
				one_small_sphere_mirrorr(97, 134, 140, 1162, 656, 6917);
				one_small_sphere_mirrorr(109, 211, 143, -70, 523, 7349);
				one_small_sphere_mirrorr(148, 46, 40, 36, 509, 7379);
				one_small_sphere_mirrorr(207, 178, 214, 1045, 1204, 5550);
				one_small_sphere_mirrorr(91, 30, 217, -40, 905, 6213);
				one_small_sphere_mirrorr(166, 203, 111, 1109, 272, 8568);
				one_small_sphere_mirrorr(62, 222, 63, 542, 404, 7757);
				one_small_sphere_mirrorr(21, 248, 159, 1316, 894, 6291);
				one_small_sphere_mirrorr(79, 236, 52, -463, 1200, 5631);
				one_small_sphere_mirrorr(49, 236, 183, 811, 497, 7401);
				one_small_sphere_mirrorr(226, 186, 97, 1040, 465, 7566);
				one_small_sphere_mirrorr(128, 19, 231, 471, 259, 8527);
				one_small_sphere_mirrorr(47, 181, 23, 118, 323, 8184);
				one_small_sphere_mirrorr(39, 197, 56, -248, 464, 7625);
				one_small_sphere_mirrorr(64, 36, 233, 1140, 815, 6459);
				one_small_sphere_mirrorr(236, 98, 191, 493, 648, 6872);
				one_small_sphere_mirrorr(120, 57, 105, 968, 1202, 5546);
				one_small_sphere_mirrorr(109, 126, 25, 1360, 1047, 5938);
				one_small_sphere_mirrorr(2, 180, 31, 1185, 1050, 5898);
				one_small_sphere_mirrorr(229, 63, 115, 1350, 541, 7360);
				one_small_sphere_mirrorr(212, 75, 49, 374, 381, 7862);
				one_small_sphere_mirrorr(189, 104, 237, 930, 426, 7705);
				one_small_sphere_mirrorr(121, 112, 251, 154, 884, 6243);
				one_small_sphere_mirrorr(149, 87, 3, 650, 227, 8771);
				one_small_sphere_mirrorr(139, 25, 36, 474, 269, 8460);
				one_small_sphere_mirrorr(229, 51, 144, 387, 517, 7313);
				one_small_sphere_mirrorr(106, 29, 239, 485, 456, 7540);
				one_small_sphere_mirrorr(2, 202, 223, 856, 1233, 5473);
				one_small_sphere_mirrorr(254, 67, 210, 1065, 752, 6617);
				one_small_sphere_mirrorr(47, 12, 17, -516, 1166, 5713);
				one_small_sphere_mirrorr(108, 101, 83, 322, 426, 7668);
				one_small_sphere_mirrorr(38, 165, 221, 1005, 1102, 5758);
				one_small_sphere_mirrorr(204, 160, 192, 1266, 385, 7984);
				one_small_sphere_mirrorr(177, 32, 70, 1334, 273, 8680);
				one_small_sphere_mirrorr(0, 65, 36, 756, 1147, 5641);
				one_small_sphere_mirrorr(195, 38, 12, 1058, 320, 8248);
				one_small_sphere_mirrorr(187, 169, 28, -381, 667, 6940);
				one_small_sphere_mirrorr(247, 80, 216, 193, 257, 8576);
				one_small_sphere_mirrorr(50, 222, 59, 117, 257, 8589);
				one_small_sphere_mirrorr(116, 224, 240, -234, 1135, 5720);
				one_small_sphere_mirrorr(141, 126, 236, -193, 359, 8084);
				one_small_sphere_mirrorr(38, 30, 101, -117, 1028, 5936);
				one_small_sphere_mirrorr(122, 236, 17, 1309, 292, 8525);
				one_small_sphere_mirrorr(221, 54, 206, 1280, 259, 8748);
				one_small_sphere_mirrorr(117, 131, 225, 1334, 956, 6143);
				one_small_sphere_mirrorr(168, 126, 128, 438, 440, 7606);
				one_small_sphere_mirrorr(52, 17, 64, -330, 1116, 5780);
				one_small_sphere_mirrorr(141, 185, 113, 432, 1193, 5540);
				one_small_sphere_mirrorr(45, 244, 42, 894, 265, 8538);
				one_small_sphere_mirrorr(140, 60, 37, 1197, 406, 7861);
				one_small_sphere_mirrorr(218, 109, 102, -498, 664, 6984);
				one_small_sphere_mirrorr(94, 235, 187, 1336, 1310, 5386);
				one_small_sphere_mirrorr(253, 123, 112, 1217, 328, 8263);
				one_small_sphere_mirrorr(3, 1, 130, 787, 657, 6858);
				one_small_sphere_mirrorr(230, 239, 8, -272, 1367, 5266);
				one_small_sphere_mirrorr(245, 105, 84, 414, 275, 8423);
				one_small_sphere_mirrorr(48, 41, 224, 507, 238, 8680);
				one_small_sphere_mirrorr(181, 56, 153, -8, 865, 6309);
				one_small_sphere_mirrorr(119, 227, 146, 1283, 709, 6787);
				one_small_sphere_mirrorr(162, 124, 194, 208, 1198, 5539);
				one_small_sphere_mirrorr(173, 83, 230, 562, 237, 8684);
				one_small_sphere_mirrorr(111, 226, 20, 144, 466, 7526);
				one_small_sphere_mirrorr(160, 234, 60, 576, 305, 8245);
				one_small_sphere_mirrorr(72, 40, 213, 1291, 1071, 5869);
				one_small_sphere_mirrorr(225, 67, 248, 956, 315, 8248);
				one_small_sphere_mirrorr(198, 9, 96, 801, 1097, 5749);
				one_small_sphere_mirrorr(19, 232, 55, 39, 1191, 5567);
				one_small_sphere_mirrorr(146, 157, 153, 279, 949, 6076);
				one_small_sphere_mirrorr(10, 173, 31, 313, 720, 6665);
				one_small_sphere_mirrorr(186, 231, 110, 237, 501, 7382);
				one_small_sphere_mirrorr(113, 247, 16, 1057, 627, 6989);
				one_small_sphere_mirrorr(23, 180, 200, 391, 808, 6421);
				one_small_sphere_mirrorr(140, 95, 20, 517, 1004, 5944);
				one_small_sphere_mirrorr(115, 3, 19, 531, 1346, 5244);
				one_small_sphere_mirrorr(3, 125, 45, 91, 245, 8687);
				one_small_sphere_mirrorr(175, 86, 113, 528, 227, 8765);
				one_small_sphere_mirrorr(10, 162, 39, 223, 254, 8591);
				one_small_sphere_mirrorr(173, 179, 116, -19, 732, 6668);
				one_small_sphere_mirrorr(156, 108, 65, 70, 587, 7103);
				one_small_sphere_mirrorr(163, 111, 59, -293, 1152, 5696);
				one_small_sphere_mirrorr(188, 195, 198, 634, 391, 7819);
				one_small_sphere_mirrorr(235, 248, 14, 732, 867, 6277);
				one_small_sphere_mirrorr(109, 79, 26, 813, 1293, 5353);
				one_small_sphere_mirrorr(116, 96, 137, 799, 249, 8626);
				one_small_sphere_mirrorr(219, 40, 72, 355, 1043, 5858);
				one_small_sphere_mirrorr(238, 119, 28, 1003, 392, 7871);
				one_small_sphere_mirrorr(145, 41, 187, -69, 1367, 5238);
				one_small_sphere_mirrorr(226, 0, 105, 968, 709, 6726);
				one_small_sphere_mirrorr(19, 120, 107, 860, 513, 7351);
				one_small_sphere_mirrorr(174, 132, 138, 766, 355, 8001);
				one_small_sphere_mirrorr(138, 199, 20, 1310, 1385, 5240);
				one_small_sphere_mirrorr(129, 25, 131, 846, 1343, 5261);
				one_small_sphere_mirrorr(196, 69, 159, -99, 387, 7915);
				one_small_sphere_mirrorr(178, 185, 48, 249, 1343, 5255);
				one_small_sphere_mirrorr(130, 146, 195, -76, 1055, 5869);
				one_small_sphere_mirrorr(91, 16, 253, 1349, 1134, 5745);
				one_small_sphere_mirrorr(43, 160, 132, 537, 1341, 5253);
				one_small_sphere_mirrorr(30, 203, 254, 206, 1347, 5251);
				one_small_sphere_mirrorr(131, 114, 212, 824, 432, 7661);
				one_small_sphere_mirrorr(30, 2, 142, -475, 386, 8067);
				one_small_sphere_mirrorr(13, 157, 70, 286, 1148, 5636);
				one_small_sphere_mirrorr(190, 47, 93, 1307, 813, 6499);
				one_small_sphere_mirrorr(0, 224, 53, 280, 762, 6550);
				one_small_sphere_mirrorr(192, 147, 37, 528, 259, 8528);
				one_small_sphere_mirrorr(145, 171, 171, -5, 269, 8548);
				one_small_sphere_mirrorr(240, 93, 93, -296, 431, 7779);
				one_small_sphere_mirrorr(214, 175, 2, 1085, 263, 8617);
				one_small_sphere_mirrorr(134, 78, 103, 1010, 1302, 5354);
				one_small_sphere_mirrorr(123, 67, 141, -263, 1135, 5727);
				one_small_sphere_mirrorr(188, 24, 15, -482, 1016, 6035);
				one_small_sphere_mirrorr(173, 176, 204, 67, 1203, 5541);
				one_small_sphere_mirrorr(114, 161, 114, 825, 499, 7398);
				one_small_sphere_mirrorr(196, 207, 15, 905, 972, 6038);
				one_small_sphere_mirrorr(101, 159, 144, 700, 292, 8331);
				one_small_sphere_mirrorr(210, 49, 253, 231, 1220, 5493);
				one_small_sphere_mirrorr(132, 7, 25, 866, 1242, 5457);
				one_small_sphere_mirrorr(228, 93, 64, 1305, 537, 7360);
				one_small_sphere_mirrorr(220, 55, 33, 134, 787, 6494);
				one_small_sphere_mirrorr(126, 249, 130, -83, 826, 6422);
				one_small_sphere_mirrorr(191, 88, 139, -380, 285, 8616);
				one_small_sphere_mirrorr(75, 222, 136, 522, 1267, 5393);
				one_small_sphere_mirrorr(209, 206, 218, 663, 658, 6845);
				one_small_sphere_mirrorr(157, 84, 206, -450, 1280, 5467);
				one_small_sphere_mirrorr(232, 187, 161, -14, 461, 7576);
				one_small_sphere_mirrorr(68, 36, 131, 448, 899, 6190);
				one_small_sphere_mirrorr(201, 216, 220, 1103, 1331, 5309);
				one_small_sphere_mirrorr(137, 63, 125, 37, 902, 6211);
				one_small_sphere_mirrorr(252, 221, 53, 74, 1096, 5763);
				one_small_sphere_mirrorr(89, 220, 56, -269, 720, 6751);
				one_small_sphere_mirrorr(252, 4, 121, 549, 545, 7211);
				one_small_sphere_mirrorr(205, 64, 16, 365, 983, 5994);
				one_small_sphere_mirrorr(64, 191, 84, -232, 710, 6771);
				one_small_sphere_mirrorr(234, 161, 185, 726, 1219, 5493);
				one_small_sphere_mirrorr(104, 100, 198, -515, 442, 7822);
				one_small_sphere_mirrorr(230, 30, 138, -325, 621, 7071);
				one_small_sphere_mirrorr(108, 223, 29, 787, 411, 7747);
				one_small_sphere_mirrorr(4, 118, 182, 1194, 893, 6270);
				one_small_sphere_mirrorr(199, 84, 152, 1100, 958, 6094);
				one_small_sphere_mirrorr(145, 216, 250, -35, 270, 8550);
				one_small_sphere_mirrorr(101, 124, 27, 503, 647, 6876);
				one_small_sphere_mirrorr(186, 200, 123, -368, 502, 7511);
				one_small_sphere_mirrorr(108, 130, 62, 520, 357, 7971);
			}

			// Wiele ma�ych
			scene->add_sphere(
				d3(center.x, G::HEIGHT + u(-400), center.z + rad - u(3000)), small_rad,
				u(0), u(0), Surface_type::diffuse,
				RGB(255, 140, 0)
			);
		}
	}

	setup_DEFAULT_details;

	setup_END;
}
