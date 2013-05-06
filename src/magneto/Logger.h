/*
 * Copyright 2012, 2013 by the Micromagnum authors.
 *
 * This file is part of MicroMagnum.
 * 
 * MicroMagnum is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MicroMagnum is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "config.h"
#include <sstream>
#include <iostream>

//
// A simple logging facility. Invoked through macros:
//
// LOG_DEBUG << "Log entry"; // a new line is not needed
// LOG_ERROR << "Error: " << error_number;
//
class Logger
{
public:
	// from least to most severe
	enum LogLevel
	{
		LOG_LEVEL_DEBUG=0,
		LOG_LEVEL_INFO=1,
		LOG_LEVEL_WARN=2,
		LOG_LEVEL_ERROR=3,
		LOG_LEVEL_CRITICAL=4
	};

	Logger(LogLevel level);
	~Logger();

	std::ostream &log(const char *file, int line);

private:
	const LogLevel level;
	std::stringstream log_stream;
};

#define LOG_DEBUG    Logger(Logger::LOG_LEVEL_DEBUG   ).log(__FILE__, __LINE__)
#define LOG_INFO     Logger(Logger::LOG_LEVEL_INFO    ).log(__FILE__, __LINE__)
#define LOG_WARN     Logger(Logger::LOG_LEVEL_WARN    ).log(__FILE__, __LINE__)
#define LOG_ERROR    Logger(Logger::LOG_LEVEL_ERROR   ).log(__FILE__, __LINE__)
#define LOG_CRITICAL Logger(Logger::LOG_LEVEL_CRITICAL).log(__FILE__, __LINE__)

#endif
