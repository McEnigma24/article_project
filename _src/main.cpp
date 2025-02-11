#include "__preprocessor__.h"
#include "__time_stamp__.h"

int main(int argc, char* argv[])
{
    srand(time(NULL));
    UTILS::clear_terminal();
    time_stamp("It just works");

    return 0;
}
