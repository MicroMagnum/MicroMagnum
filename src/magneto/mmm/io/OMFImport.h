#ifndef OMF_IMPORT_H
#define OMF_IMPORT_H

#include "config.h"

#include "matrix/matty.h"

#include <string>
#include <istream>

#include "OMFHeader.h"

VectorMatrix readOMF(const std::string &path, OMFHeader &header);
VectorMatrix readOMF(       std::istream &in, OMFHeader &header);

#endif

