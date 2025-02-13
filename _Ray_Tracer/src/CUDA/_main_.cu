#include "RT_Renderer.h"
#include "CTRL_Setuper.h"


// CTRL     - Constrol sofware - stast, execution, changing scenes
// RT       - Ray Tracing related structures and functions
// FUNC     - Generic Math


void good_to_go(int argc, char* argv[])
{
    if(argc >= 4)
    {
        CPU_LINE(G::MODEL_NAME = argv[1]);
        GPU_LINE(G::MODEL_NAME = argv[2]);

        CPU_LINE(G::NUM_OF_CORES = argv[3]);
        GPU_LINE(G::NUM_OF_CORES = "0");

        if(argc == 6)
        {
            G::SCALING_MULTI = atof(argv[4]); 
            G::SCALING_ADD = atof(argv[5]); 
        }
    }
    else
    {
        cout << "NOT ENOUGHT PARAMS fead into main function\n"; 
        exit(0);
    }

    #ifdef GPU
        string no_gpu = "nvidia_gpu_not_present";
        if(G::MODEL_NAME == no_gpu)
        {
            cout << "Nvidia GPU not present. Exiting...\n";
            exit(0);
        }
    #endif

    ifstream FILE(path_rep_num);
    if (!FILE.is_open()) {
        std::cerr << "Nie mozna otworzyc pliku: " << path_rep_num << std::endl;
        exit(0);
    }
    FILE >> G::REP_NUMBER;

    if(G::REP_NUMBER == 0) { cout << "REP_NUMBER == 0\n"; exit(0); }
}

void macros_check()
{
    // WIN & LINUX
    #if !defined(WIN) && !defined(LINUX)
    line("ERROR - WIN - LINUX - NONE of the platform macros were defined");
    exit(0);
    #endif
    #if defined(WIN) && defined(LINUX)
    line("ERROR - WIN - LINUX - BOTH of the platform macros were defined");
    exit(0);
    #endif

    // CPU & GPU
    #if !defined(CPU) && !defined(GPU)
    line("ERROR - CPU - GPU - NONE of the platform macros were defined");
    exit(0);
    #endif
    #if defined(CPU) && defined(GPU)
    line("ERROR - CPU - GPU - BOTH of the platform macros were defined");
    exit(0);
    #endif

    #ifdef CPU
    // ARCH_X86 & ARCH_X64
    #if !defined(ARCH_X86) && !defined(ARCH_X64)
    line("ERROR - ARCH_X86 - ARCH_X64 - NONE of the platform macros were defined");
    exit(0);
    #endif
    #if defined(ARCH_X86) && defined(ARCH_X64)
    line("ERROR - ARCH_X86 - ARCH_X64 - BOTH of the platform macros were defined");
    exit(0);
    #endif
    #endif

    // UNIT_FLOAT & UNIT_DOUBLE
    #if !defined(UNIT_FLOAT) && !defined(UNIT_DOUBLE)
    line("ERROR - UNIT_FLOAT - UNIT_DOUBLE - NONE of the platform macros were defined");
    exit(0);
    #endif
    #if defined(UNIT_FLOAT) && defined(UNIT_DOUBLE)
    line("ERROR - UNIT_FLOAT - UNIT_DOUBLE - BOTH of the platform macros were defined");
    exit(0);
    #endif
}

#define exiting_app { WIN_LINE(window.close()); return 0; }
bool display_controller = false;

int main_main(int argc, char* argv[])
{
    good_to_go(argc, argv);
    macros_check();

    srand((unsigned int)time(NULL));
    Setuper::setup_Global_Variables___and___Clear_Stats();

    #ifdef LINUX
        Renderer render;

        for (;;)
        {
            render.RENDER();

            if (render.test_is_finished()) { exiting_app; }
        }

    #endif

    #ifdef WIN
        sf::RenderWindow window(sf::VideoMode(static_cast<unsigned int>(G::WIDTH), static_cast<unsigned int>(G::HEIGHT)), "Ray Tracer");
        Renderer render(window);

        #ifdef INTERACTIVE_MODE
        sf::Mouse::setPosition(Vector2i(G::WIDTH / 2, G::HEIGHT / 2), window);
        #endif

        while (window.isOpen())
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                switch (event.type)
                {
                case sf::Event::Closed:
                {
                    exiting_app;
                }

                case sf::Event::MouseWheelMoved:
                {
                    G::Camera::pos.z += u(event.mouseWheel.delta * G::Camera::move_speed);

                    /*if(event.mouseWheel.delta > 0)
                        G::Camera::rotation_angles.x -= G::Camera::rotation_speed;
                    else
                        G::Camera::rotation_angles.x += G::Camera::rotation_speed;*/



                    break;
                }

                #ifdef INTERACTIVE_MODE
                case sf::Event::MouseMoved:
                {
                    if (event.mouseMove.x == G::WIDTH / 2 && event.mouseMove.y == G::HEIGHT / 2) break;

                    if (G::WIDTH / 2 < event.mouseMove.x)
                    {
                        G::Camera::rotation_angles.x -= G::Camera::rotation_speed;
                    }
                    else
                    {
                        G::Camera::rotation_angles.x += G::Camera::rotation_speed;
                    }

                    if (G::HEIGHT / 2 < event.mouseMove.y)
                    {
                        G::Camera::rotation_angles.y -= G::Camera::rotation_speed;
                    }
                    else
                    {
                        G::Camera::rotation_angles.y += G::Camera::rotation_speed;
                    }

                    sf::Mouse::setPosition(Vector2i(G::WIDTH / 2, G::HEIGHT / 2), window);
                    break;
                }
                #endif // INTERACTIVE_MODE

                case sf::Event::MouseButtonPressed:
                {
                    if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
                    {
                        // cmp = sf::Mouse::getPosition(window); //cout << cmp.x << ", " << cmp.y << endl;

                        // buttons.click_check_buttons(cmp);
                    }

                    break;
                }

                case sf::Event::KeyPressed:
                {
                    switch (event.key.code)
                    {
                    case sf::Keyboard::Escape:
                    {
                        exiting_app;
                    }

                    case sf::Keyboard::Num1:
                    {
                        G::PIXEL_SKIP = 1;
                        render.reset_all_pixels();
                        break;
                    }
                    case sf::Keyboard::Num2:
                    {
                        G::PIXEL_SKIP = 2;
                        render.reset_all_pixels();
                        break;
                    }
                    case sf::Keyboard::Num3:
                    {
                        G::PIXEL_SKIP = 3;
                        render.reset_all_pixels();
                        break;
                    }

                    case sf::Keyboard::Numpad0:
                    {
                        display_controller = !display_controller;
                        delay_input;
                        break;
                    }

                    #ifdef INTERACTIVE_MODE
                    case sf::Keyboard::Q:
                    {
                        G::Camera::rotation_angles.z += G::Camera::rotation_speed * 10;
                        break;
                    }

                    case sf::Keyboard::E:
                    {
                        G::Camera::rotation_angles.z -= G::Camera::rotation_speed * 10;
                        break;
                    }

                    case sf::Keyboard::A:
                    {
                        G::Camera::move_offset.x -= G::Camera::move_speed;





                        // Zrobi�, �eby w zale�no�ci od rotation_angles mo�na by�o si� porusza� 
                        // po x, y, z  np. id�c tylko do przodu





                        break;
                    }

                    case sf::Keyboard::D:
                    {
                        G::Camera::move_offset.x += G::Camera::move_speed;
                        break;
                    }

                    case sf::Keyboard::W:
                    {
                        G::Camera::move_offset.z += G::Camera::move_speed * 1;
                        break;
                    }

                    case sf::Keyboard::S:
                    {
                        G::Camera::move_offset.z -= G::Camera::move_speed * 1;
                        break;
                    }

                    case sf::Keyboard::Space:
                    {
                        G::Camera::move_offset.y -= G::Camera::move_speed;
                        break;
                    }

                    case sf::Keyboard::LControl:
                    {
                        G::Camera::move_offset.y += G::Camera::move_speed;
                        break;
                    }
                    #endif // INTERACTIVE_MODE

                    default:
                    { break; }
                    }
                }

                default:
                { break; }
                }
            }

            // DISPLAY CONTROLLER //
            if (display_controller) continue;


            window.clear();
            {
                render.RENDER();
            }
            window.display();

            // Sleep(1000);


            if (render.test_is_finished()) { exiting_app; }
        }

    #endif

    return 0;
}
