#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "config.h"
#include <map>
#include <ostream>

/**
 * Benchmark class.
 */
class Benchmark
{
public:
	struct Record 
	{
		std::string id, desc;
		int num_benchmarks;
		double total_time;
		double t0;
	};

	/**
	 * Starts a benchmarked section with benchmark id.
	 *
	 * @param id the benchmark id
	 */
	void tic(const std::string &id);

	/**
	 * Stops the time for a benchmarked section with benchmark id
	 *
	 * @param id the benchmark id
	 */
	void toc(const std::string &id);

	/**
	 * Returns time in milliseconds since some fixed point in the past.
	 */
	static double now();

	/**
	 * Optionally sets a description for a benchmark id that shows up in the summary.
	 *
	 * @param id the benchmark id
	 * @param desc the descripton of the benchmark
	 *
	 */
	void setDescription(const std::string &id, const std::string &desc);

	/**
	 * Clears all previously recorded benchmark data.
	 */
	void reset();

	/**
	 * Prints a report of all recorded benchmarks runs.
	 *
	 * @param out the output stream to print the report 
	 */
	void report(std::ostream &out);

	/**
	 * Like report(std::ostream&), but accepts file name instead of stream.
	 * @param path the file to print the report (file gets overwritten)
	 */
	void reportToFile(std::string path);

	/**
	 * Returns the singleton Benchmark instance.
	 */
	static Benchmark &inst();

	Record &getRecord(const std::string &id);

private:
	std::map<std::string, Record> records;
};

#endif
