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

#include "config.h"
#include "Logger.h"
#include "Benchmark.h"

#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cstddef>

#include "os.h"

Benchmark bench;

Benchmark::Record &Benchmark::getRecord(const std::string &id)
{
	std::map<std::string, Record>::iterator it = records.find(id);
	if (it == records.end()) {
		Record r;
		r.id = id;
		r.num_benchmarks = 0;
		r.total_time = 0.0;
		records[id] = r;
		it = records.find(id);
	}
	return it->second;
}

double Benchmark::now()
{
	return os::getTickCount();
}

void Benchmark::tic(const std::string &id)
{
	getRecord(id).t0 = now();
}

void Benchmark::toc(const std::string &id)
{
	Record &record = getRecord(id);
	record.total_time += (now() - record.t0);
	record.num_benchmarks += 1;
	record.t0 = now();
}

void Benchmark::setDescription(const std::string &id, const std::string &desc)
{
	getRecord(id).desc = desc;
}

void Benchmark::reset()
{
	records.clear();
}

void Benchmark::report(std::ostream &out)
{
	// I. Get sorted list of all record ids
	std::vector<std::string> ids;
	for (std::map<std::string, Record>::iterator it=records.begin(); it!=records.end(); ++it) {
		ids.push_back(it->first);
	}
	std::sort(ids.begin(), ids.end(), std::less<std::string>());

	// II. Summarize
	out << "                                (average time)  (total time)     (# evals)" << std::endl;
	out << "==========================================================================" << std::endl;

	for (std::vector<std::string>::iterator it=ids.begin(); it!=ids.end(); ++it) {
		const std::string &id = *it;		
		Record &record = getRecord(id);

		const double avg_time = record.total_time / double(record.num_benchmarks);
		const std::string &desc = (record.desc == "") ? *it : record.desc;
	
		// Count number of "." separator occurences for indenting
		int indent = 0;
		for (size_t i=0; i<id.length(); ++i) {
			if (id[i] == '.') indent += 1;
		}

		out << std::setw(indent*2) << "";
		out << std::left  << std::setw(30-indent*2) << std::setfill('.') << desc << std::setfill(' ') << ": ";
		out << std::right << std::setw(14) << avg_time;
		out << std::right << std::setw(14) << record.total_time;
		out << std::right << std::setw(14) << record.num_benchmarks;
		out << std::endl;
	}
}

void Benchmark::reportToFile(std::string path)
{
	std::ofstream out(path.c_str());
	report(out);
}

Benchmark &Benchmark::inst()
{
	return ::bench;
}
