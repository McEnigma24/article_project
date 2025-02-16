#pragma once

#define SHORTCUT_LOGS 1

// #define INTERACTIVE_MODE

#ifdef WIN
#define WIN_LINE(...) __VA_ARGS__
#else
#define WIN_LINE(...)
#endif

#ifdef LINUX
#define LINUX_LINE(...) __VA_ARGS__
#else
#define LINUX_LINE(...)
#endif

#ifdef CPU
#define CPU_LINE(...) __VA_ARGS__
#else
#define CPU_LINE(...)
#endif

#ifdef GPU
#define GPU_LINE(...) __VA_ARGS__
#else
#define GPU_LINE(...)
#endif

#ifdef ARCH_X86
#define ARCH_X86_LINE(...) __VA_ARGS__
#else
#define ARCH_X86_LINE(...)
#endif

#ifdef ARCH_X64
#define ARCH_X64_LINE(...) __VA_ARGS__
#else
#define ARCH_X64_LINE(...)
#endif

#ifdef UNIT_FLOAT
#define UNIT_FLOAT_LINE(...) __VA_ARGS__
#else
#define UNIT_FLOAT_LINE(...)
#endif

#ifdef UNIT_DOUBLE
#define UNIT_DOUBLE_LINE(...) __VA_ARGS__
#else
#define UNIT_DOUBLE_LINE(...)
#endif

#define SCENE_MAX_NUM 50

#define NEW
