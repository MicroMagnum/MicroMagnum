#include "config.h"
#include "OMFHeader.h"

OMFHeader::OMFHeader()
	: Title("<title>"),
	  meshunit("<meshunit>"),
	  valueunit("<valueunit>"),
	  valuemultiplier(1.0),
	  xmin(0.0), ymin(0.0), zmin(0.0),
	  xmax(0.0), ymax(0.0), zmax(0.0),
	  ValueRangeMaxMag(0.0), ValueRangeMinMag(0.0),
	  meshtype("rectangular"),
	  xbase(0.0), ybase(0.0), zbase(0.0),
	  xstepsize(0.0), ystepsize(0.0), zstepsize(0.0),
	  xnodes(0), ynodes(0), znodes(0)
{
}

OMFHeader::~OMFHeader()
{
}
