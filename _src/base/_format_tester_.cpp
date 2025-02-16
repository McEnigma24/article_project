#include <iostream>

#define MACRO_ONE_LINE(x) std::cout << #x << " = " << x << "\n"
#define MACRO_MULTI_LINE(x)                                                    \
    std::cout << #x << " = " << x;                                             \
    std::cout << "\n"

namespace TestNamespace
{
    void namespaceFunction()
    {
        std::cout << "Inside namespace function" << std::endl;
    }
} // namespace TestNamespace
class TestClass
{
  public:
    TestClass() { std::cout << "Constructor" << std::endl; }

    void memberFunction() { std::cout << "Member function" << std::endl; }
};

struct TestStruct
{
    int member;
};

union TestUnion
{
    int intMember;
    float floatMember;
};

enum TestEnum
{
    Value1,
    Value2
};

void freeFunction()
{
    std::cout << "Free 1 function" << std::endl;
    std::cout << "Free 2 function" << std::endl;
}

int main_test()
{
    TestNamespace::namespaceFunction();

    TestClass obj;
    obj.memberFunction();

    TestStruct s;
    s.member = 10;

    TestUnion u;
    u.intMember = 20;

    TestEnum e = Value1;

    if (e == Value1)
    {
        std::cout << "Enum is Value1" << std::endl;
    }
    else
    {
        std::cout << "Enum is not Value1" << std::endl;
    }

    for (int i = 0; i < 5; ++i)
    {
        std::cout << "Loop iteration " << i << std::endl;
    }

    while (false)
    {
        std::cout << "This will not print" << std::endl;
    }

    try
    {
        throw std::runtime_error("Exception");
    }
    catch (const std::exception &ex)
    {
        std::cout << "Caught exception: " << ex.what() << std::endl;
    }

    return 0;
}