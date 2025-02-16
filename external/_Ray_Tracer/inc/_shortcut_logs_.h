#pragma once
#include "_pre_code_mode_.h"
#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#if SHORTCUT_LOGS == 1

#define print_var_name cout << var_name << " -> ";
#define call_print(x) x.print(#x);

#define perc(x) cout << #x << " = " << std::setprecision(2) << std::fixed << static_cast<float>(x) << "%\n";
#define percc(x) cout << #x << " = " << std::setprecision(2) << std::fixed << static_cast<float>(x) << "% ";
#define var(x) cout << #x << " = " << x << '\n';
#define varr(x) cout << #x << " = " << x << ' ';
#define coord(x, y) cout << "(" << x << ", " << y << ")" << '\n';
#define line(x) cout << x << '\n';
#define linee(x) cout << x << ' ';
#define nline cout << '\n';
#define my_loge cout << "LOG: " << __FILENAME__ << " " << __LINE__ << "   \n";
#define my_log(x) cout << "LOG: " << __FILENAME__ << " " << __LINE__ << "   " << x << "\n";
#define my_logg(x) cout << "LOG: " << __FILENAME__ << " " << __LINE__ << "   " << x << " ";

#define var_tuple(x)                                                                                                   \
    {                                                                                                                  \
        auto [a, b, c, d] = x.get();                                                                                   \
        cout << #x << " = ( " << std::to_string(a) << ", " << std::to_string(b) << ", " << std::to_string(c) << ", "   \
             << std::to_string(d) << " )\n";                                                                           \
    }

#else

#define print_var_name
#define call_print(x)

#define perc(x)
#define percc(x)
#define var(x)
#define varr(x)
#define coord(x, y)
#define line(x)
#define linee(x)
#define nline

#define var_tuple(x)

#endif