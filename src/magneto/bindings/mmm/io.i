%{
#include "mmm/io/OMFExport.h"
#include "mmm/io/OMFImport.h"
#include "mmm/io/OMFHeader.h"
%}

enum OMFFormat 
{
	OMF_FORMAT_ASCII=0,
	OMF_FORMAT_BINARY_4=1,
	OMF_FORMAT_BINARY_8=2
};

struct OMFHeader 
{
	OMFHeader();
	~OMFHeader();

	std::string Title;
	std::vector<std::string> Desc;
	std::string meshunit;
	std::string valueunit;
	double valuemultiplier;
	double xmin, ymin, zmin;
	double xmax, ymax, zmax;
	double ValueRangeMaxMag, ValueRangeMinMag;
	std::string meshtype;
	double xbase, ybase, zbase;
	double xstepsize, ystepsize, zstepsize;
	int xnodes, ynodes, znodes;
};

VectorMatrix  readOMF(const std::string &path, OMFHeader &header);
void         writeOMF(const std::string &path, OMFHeader &header, const VectorMatrix &field, OMFFormat format);

