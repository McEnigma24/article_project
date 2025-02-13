#include "RT_Dimensions.h"

#ifdef CPU
void d1::print(const string& var_name) const
{
	print_var_name;
	var(x);
}

void d2::print(const string& var_name) const
{
	print_var_name;
	varr(x);
	var(y);
}

void d3::print(const string& var_name) const
{
	print_var_name;
	varr(x);
	varr(y);
	var(z);
}
#endif
