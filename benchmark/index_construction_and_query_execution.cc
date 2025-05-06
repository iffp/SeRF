/**
 * @file benchmark_arbitrary.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Arbitrary Range Filter Search
 * @date 2024-11-17
 *
 * @copyright Copyright (c) 2024
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
// #include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

// FANNS survey
#include <chrono>
#include <thread>
#include "fanns_survey_helpers.cpp"

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

int main(int argc, char **argv) {
	#ifdef USE_SSE
	  cout << "Use SSE" << endl;
	#endif
	#ifdef USE_AVX
	  cout << "Use AVX" << endl;
	#endif
	#ifdef USE_AVX512
	  cout << "Use AVX512" << endl;
	#endif
	#ifndef NO_PARALLEL_BUILD
	  cout << "Index Construct Parallelly" << endl;
	#endif
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "Number of threads: " << nthreads << std::endl;

	// Parameters
	string path_database_vectors;
	string path_database_attributes;
	string path_query_vectors;
	string path_query_attributes;
	string path_groundtruth;
	int index_k;
	int ef_construction;
	int ef_max;
	vector<int> ef_search_list;
	int k;

    // Parse arguments
    if (argc != 11) {
        fprintf(stderr, "Usage: %s <path_database_vectors> <path_database_attributes> <path_query_vectors> <path_query_attributes> <path_groundtruth> <index_k> <ef_construction> <ef_max> <ef_search_list> <k>\n", argv[0]);
        exit(1);
    }

	// Store parameters
	path_database_vectors = argv[1];
	path_database_attributes = argv[2];
	path_query_vectors = argv[3];
	path_query_attributes = argv[4];
	path_groundtruth = argv[5];
	index_k = atoi(argv[6]);
	ef_construction = atoi(argv[7]);
	ef_max = atoi(argv[8]);
	ef_search_list = parse_int_list(argv[9]);
	k = atoi(argv[10]);


	
    // Load database vectors
    vector<vector<float>> database_vectors = read_fvecs(path_database_vectors);
    int n_items = database_vectors.size();
    int d = database_vectors[0].size();

    // Load database attributes
    vector<int> database_attributes = read_one_int_per_line(path_database_attributes);
    assert(database_attributes.size() == n_items);

	// Load query vectors
	vector<vector<float>> query_vectors = read_fvecs(path_query_vectors);
	int n_queries = query_vectors.size();
	assert(query_vectors[0].size() == d);

	// Load query attributes
	vector<pair<int, int>> query_attributes = read_two_ints_per_line(path_query_attributes);
	assert(query_attributes.size() == n_queries);

	// Read ground truth
    vector<vector<int>> groundtruth = read_ivecs(path_groundtruth);
    assert(groundtruth.size() == n_queries);

    // Truncate ground-truth to at most k items
	// How does SeRF behave if we have less than k matches for a given query? -> Doesn't seem to be a problem
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

	// Sort the database by attribute value and remap the groundtruth accordingly
	// Withouth this step, SeRF doesn't work
	// TODO: Should this step also be timed for benchmarking?
	sort_by_attribute_and_remap(database_vectors, database_attributes, groundtruth);

	// Initialize a data wrapper (needed by SeRF)
	// query_num and query_k are not used in in index construction as they are only needed during query execution
	// the dataset parameter is also not used as we manually load the data
	// NOTE: querys_keys doesn't seem to appear anywhere in the code except for /include/common/data_wrapper.h
	// NOTE: query_ids seems to allow some reordering of query id's but I don't think it is needed.
	DataWrapper data_wrapper(n_queries, k, "custom", n_items);
	data_wrapper.version = "Benchmark";		// Is this used? I don't think so
	data_wrapper.data_dim = d;				
	data_wrapper.real_keys = false;			// Is this used? I don't think so
	data_wrapper.nodes = database_vectors;	
	data_wrapper.nodes_keys = database_attributes;	
	data_wrapper.querys = query_vectors;
	data_wrapper.query_ranges = query_attributes;	
	data_wrapper.groundtruth = groundtruth;	
	
	// Initialize and configure the index
	BaseIndex::IndexParams i_params(index_k, ef_construction, ef_construction, ef_max);
	i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
	base_hnsw::L2Space ss(d);
	SeRF::IndexSegmentGraph2D serf_index(&ss, &data_wrapper);
	BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D", "benchmark");

	// Construct the index (timed)
	auto start_time = std::chrono::high_resolution_clock::now();
    serf_index.buildIndex(&i_params);
	auto end_time = std::chrono::high_resolution_clock::now();

    // Compute duration 
    std::chrono::duration<double> diff = end_time - start_time;
    double index_construction_time = diff.count();

	// Configure search
	BaseIndex::SearchParams s_params;
	s_params.query_K = k;

	// Iterate over search parameters	
	vector<double> recall_list;
	vector<double> qps_list;
	for (int ef_search : ef_search_list) {
		s_params.search_ef = ef_search;
		vector<vector<int>> results;
		// Time the query execution
		auto start_time = std::chrono::high_resolution_clock::now();
		// Iterate through queries
		for (int i = 0; i < n_queries; i++) {
			s_params.query_range = query_attributes[i].second - query_attributes[i].first + 1;	
			auto res = serf_index.rangeFilteringSearchOutBound(&s_params, &search_info, query_vectors[i], query_attributes[i]);
			results.push_back(res);
		}
		auto end_time = std::chrono::high_resolution_clock::now();

		// Compute timing 	
		std::chrono::duration<double> diff = end_time - start_time;
		double query_execution_time = diff.count();

		// Compute recall
		size_t match_count = 0;
		size_t total_count = 0;
		for (int i = 0; i < n_queries; i++){
			int n_valid_neighbors = std::min(k, (int)groundtruth[i].size());
			vector<int> groundtruth_q = groundtruth[i];
			vector<int> result_q = results[i];
			sort(groundtruth_q.begin(), groundtruth_q.end());
			sort(result_q.begin(), result_q.end());
			vector<int> intersection;
			set_intersection(groundtruth_q.begin(), groundtruth_q.end(), result_q.begin(), result_q.end(), back_inserter(intersection));
			match_count += intersection.size();
			total_count += n_valid_neighbors;
		}
		double recall = (double)match_count / total_count;
		double qps = n_queries / query_execution_time;
		recall_list.push_back(recall);
		qps_list.push_back(qps);
	}

	// Report results   
	peak_memory_footprint();
	printf("Index construction time: %.3f s\n", index_construction_time);
	for (int i = 0; i < ef_search_list.size(); i++) {
		printf("ef_search: %d QPS: %.3f Recall: %.3f\n", ef_search_list[i], qps_list[i], recall_list[i]);
	}
	return 0;
}
