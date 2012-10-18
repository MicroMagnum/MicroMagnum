#ifndef RESTRICT_H_INCLUDED
#define RESTRICT_H_INCLUDED

// GNU C++ extension for the C99 'restrict' keyword.
// The Intel compiler also supports this extension.
// The Visual C++ compiler does too.
#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#endif
