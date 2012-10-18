#include "config.h"
#include "Logger.h"
#include "Magneto.h"

Logger::Logger(LogLevel level) : level(level)
{
}

Logger::~Logger()
{
	callDebugFunction(level, log_stream.str());
}

std::ostream &Logger::log(const char *file, int lineno)
{
	return log_stream;
}

