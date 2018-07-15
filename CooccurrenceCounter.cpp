// CooccurrenceCounter.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ThreadPool.hpp"
#include "cxxopts.hpp"
#include "utils.h"

using namespace std;


string selectField(string line, size_t field)
{
	if (field == (size_t)-1) return line;
	string f;
	stringstream ts{ line };
	size_t i = 0;
	for (; i < field; ++i)
	{
		if (!getline(ts, f, '\t')) break;
	}
	if (i < field || !getline(ts, f, '\t'))
	{
		return {};
	}
	return f;
}


inline size_t makeTriangleIndex(int a, int b, size_t N)
{
	if (a > b) swap(a, b);
	return (a * (2 * N - (a + 3))) / 2 - 1 + b;
}

inline pair<int, int> decomposeTriangleIndex(size_t idx, size_t N)
{
	int i = N - 2 - int(sqrt(-8 * idx + 4 * N*(N - 1) - 7) / 2.0 - 0.5);
	int j = idx + i + 1 - N * (N - 1) / 2 + (N - i)*((N - i) - 1) / 2;
	return { i, j };
}

void countCooc(vector<uint32_t>& counter, size_t nVocab, const vector<int>& ids)
{
	for (size_t i = 0; i < ids.size(); i++) for (size_t j = i + 1; j < ids.size(); j++)
	{
		auto a = ids[i], b = ids[j];
		if (a == b) continue;
		++counter[makeTriangleIndex(a, b, nVocab)];
	}
}

struct Args
{
	string input, output, model;
	int field = 0;
	size_t maxline = -1;
	int worker = thread::hardware_concurrency();
	size_t threshold = 100;
	string pmi;
};

void cooc(const Args& args)
{
	typedef unordered_map<size_t, size_t> LDFreqCount;
	typedef vector<uint32_t> LDCoocCount;

	WordDictionary rdict;
	size_t nVocab = 0;
	ifstream infile{ args.input };
	cerr << "Scanning..." << endl;
	{
		WordDictionary dict;
		auto fcnt = scanText<LDFreqCount>(infile, args.worker, args.maxline, [&dict, &args](LDFreqCount& ld, string line, size_t numLine)
		{
			auto f = selectField(line, args.field);
			if (f.empty())
			{
				cerr << "Line " << numLine << ": no field..." << endl;
				return;
			}
			stringstream ss{ f };
			unordered_set<string> words{ istream_iterator<string>{ss}, istream_iterator<string>{} };
			if (words.empty()) return;
			for(auto&& wId : dict.getOrAdds(words.begin(), words.end())) ++ld[wId];
		});
		auto&& freq = fcnt[0];
		for (size_t i = 1; i < fcnt.size(); ++i)
		{
			for (auto&& p : fcnt[i])
			{
				freq[p.first] += p.second;
			}
		}

		for (auto&& p : freq)
		{
			if (p.second < args.threshold) continue;
			rdict.add(dict.getStr(p.first));
			++nVocab;
		}
	}
	
	cerr << "Counting..." << endl;
	infile.clear();
	infile.seekg(0);
	auto fcnt = scanText<LDCoocCount>(infile, args.worker, args.maxline, [&rdict, &args, &nVocab](LDCoocCount& ld, string line, size_t numLine)
	{
		auto f = selectField(line, args.field);
		if (f.empty())
		{
			cerr << "Line " << numLine << ": no field..." << endl;
			return;
		}
		stringstream ss{ f };
		vector<int> wids;
		for (auto w : unordered_set<string>{ istream_iterator<string>{ss}, istream_iterator<string>{} })
		{
			auto id = rdict.get(w);
			if (id < 0) continue;
			wids.emplace_back(id);
		}
		countCooc(ld, nVocab, wids);
	}, vector<uint32_t>(nVocab * (nVocab - 1) / 2));

	cerr << "Merging..." << endl;

	ThreadPool pool(args.worker);
	vector<future<void>> futures;
	futures.reserve(args.worker);
	for (size_t i = 1; i < fcnt.size(); i <<= 1)
	{
		for (size_t j = 0; j + i < fcnt.size(); j += i * 2)
		{
			futures.emplace_back(pool.enqueue([&fcnt, nVocab, i, j](size_t tId)
			{
				for (size_t n = 0; n < nVocab * (nVocab - 1) / 2; ++n)
				{
					fcnt[j][n] += fcnt[j + i][n];
				}
			}));
		}
		for (auto&& p : futures) p.wait();
		futures.clear();
	}
	auto&& counter = fcnt[0];
	vector<pair<size_t, size_t>> res;
	for (size_t n = 0; n < nVocab * (nVocab - 1) / 2; ++n)
	{
		if (counter[n] < args.threshold) continue;
		res.emplace_back(n, counter[n]);
	}
	cerr << "Sorting..." << endl;
	sort(res.begin(), res.end(), [&counter](const auto& a, const auto& b)
	{
		return a.second > b.second;
	});
	cerr << "Writing..." << endl;

	auto writeFunc = [&res, &rdict, nVocab](ostream& out)
	{
		for (auto && p : res)
		{
			auto ab = decomposeTriangleIndex(p.first, nVocab);
			out << rdict.getStr(ab.first) << '\t' << rdict.getStr(ab.second) << '\t' << p.second << endl;
		}
	};
	if (args.output.empty())
	{
		writeFunc(cout);
	}
	else
	{
		ofstream out{ args.output };
		writeFunc(out);
	}
}

void pmi(const Args& args)
{
	struct LD
	{
		vector<uint32_t> freq;
		size_t nDocs = 0;
	};
	WordDictionary rdict;
	vector<uint32_t> wordFreq;
	size_t totDocs = 0;
	size_t nVocab = 0;
	ifstream infile{ args.input };
	cerr << "Scanning..." << endl;
	{
		WordDictionary dict;
		auto fcnt = scanText<LD>(infile, args.worker, args.maxline, [&dict, &args](LD& ld, string line, size_t numLine)
		{
			auto f = selectField(line, args.field);
			if (f.empty())
			{
				cerr << "Line " << numLine << ": no field..." << endl;
				return;
			}
			istringstream ss{ f };
			unordered_set<string> words{ istream_iterator<string>{ss}, istream_iterator<string>{} };
			if (words.empty()) return;
			auto ids = dict.getOrAdds(words.begin(), words.end());
			size_t maxId = *max_element(ids.begin(), ids.end());
			if (maxId >= ld.freq.size()) ld.freq.resize(maxId + 1);
			auto wit = words.begin();
			for (auto id : ids)
			{
				if (wit++->size() <= 1) continue;
				++ld.freq[id];
			}
			++ld.nDocs;
		});
		auto&& freq = fcnt[0].freq;
		totDocs = fcnt[0].nDocs;
		for (size_t i = 1; i < fcnt.size(); ++i)
		{
			if (freq.size() < fcnt[i].freq.size()) freq.resize(fcnt[i].freq.size());
			auto it = freq.begin();
			for (auto&& p : fcnt[i].freq)
			{
				*it++ += p;
			}
			totDocs += fcnt[i].nDocs;
		}

		vector<pair<size_t, size_t>> countSorted;
		for (size_t i = 0; i < freq.size(); ++i)
		{
			countSorted.emplace_back(i, freq[i]);
		}
		sort(countSorted.begin(), countSorted.end(), [](const auto& a, const auto& b)
		{
			return a.second > b.second;
		});
		for (auto&& p : countSorted)
		{
			if (p.second < args.threshold) break;
			rdict.add(dict.getStr(p.first));
			wordFreq.emplace_back(p.second);
			++nVocab;
		}
	}
	cerr << "Counting..." << endl;
	infile.clear();
	infile.seekg(0);
	auto fcnt = scanText<vector<uint32_t>>(infile, args.worker, args.maxline, [&rdict, &args, &nVocab](vector<uint32_t>& ld, string line, size_t numLine)
	{
		auto f = selectField(line, args.field);
		if (f.empty())
		{
			cerr << "Line " << numLine << ": no field..." << endl;
			return;
		}
		istringstream ss{ f };
		vector<int> wids;
		for (auto w : unordered_set<string>{ istream_iterator<string>{ss}, istream_iterator<string>{} })
		{
			auto id = rdict.get(w);
			if (id < 0) continue;
			wids.emplace_back(id);
		}
		countCooc(ld, nVocab, wids);

	}, vector<uint32_t>(nVocab * (nVocab - 1) / 2));

	cerr << "Merging..." << endl;
	ThreadPool pool(args.worker);
	vector<future<void>> futures;
	{
		futures.reserve(args.worker);
		for (size_t i = 1; i < fcnt.size(); i <<= 1)
		{
			for (size_t j = 0; j + i < fcnt.size(); j += i * 2)
			{
				futures.emplace_back(pool.enqueue([&fcnt, nVocab, i, j](size_t tId)
				{
					for (size_t n = 0; n < nVocab * (nVocab - 1) / 2; ++n)
					{
						fcnt[j][n] += fcnt[j + i][n];
					}
				}));
			}
			for (auto&& p : futures) p.wait();
			futures.clear();
		}
	}

	auto&& counter = fcnt[0];
	vector<float>& counterF = *(vector<float>*)&counter;
	futures.resize(args.worker * 4);
	for (size_t i = 0; i < args.worker * 4; ++i)
	{
		futures[i % futures.size()] = pool.enqueue([&](size_t tId, size_t b, size_t e)
		{
			for (size_t j = b; j < e; ++j)
			{
				auto ab = decomposeTriangleIndex(j, nVocab);
				counterF[j] = log((counter[j] ? counter[j] : 0.1f) * (float)totDocs / wordFreq[ab.first] / wordFreq[ab.second]);
			}
		}, i * counter.size() / args.worker / 4, (i + 1) * counter.size() / args.worker / 4);
	}
	for (auto&& p : futures) p.wait();
	futures.clear();

	cout << "Vocab size: " << nVocab << endl;

	ofstream out{ args.output, ios::binary };
	out.write("CPMI", 4);
	rdict.writeToFile(out);
	for (auto f : counterF)
	{
		int16_t s = max(min(int(f * 1024), 32767), -32768);
		out.write((const char*)&s, 2);
	}
}

struct PMIData
{
	WordDictionary dict;
	vector<int16_t> pmis;
	size_t nVocab = 0;
};

PMIData loadPMI(const string& inputPath)
{
	PMIData d;
	ifstream input{ inputPath, ios::binary };
	char buf[4];
	input.read(buf, 4);
	if (string{ buf, buf + 4 } != "CPMI") throw exception();
	d.dict.readFromFile(input);
	d.nVocab = d.dict.size();
	d.pmis.resize(d.nVocab * (d.nVocab - 1) / 2);
	input.read((char*)&d.pmis[0], sizeof(uint16_t) * d.pmis.size());
	return d;
}

void pmiShow(const Args& args)
{
	PMIData p = loadPMI(args.model);
	vector<size_t> order;
	for (size_t i = 0; i < p.pmis.size(); ++i)
	{
		if (p.pmis[i] <= 0) continue;
		order.emplace_back(i);
	}
	sort(order.begin(), order.end(), [&p](auto a, auto b)
	{
		return p.pmis[a] > p.pmis[b];
	});

	auto printResult = [&](ostream& str)
	{
		for (auto o : order)
		{
			auto ab = decomposeTriangleIndex(o, p.nVocab);
			str << p.dict.getStr(ab.first) << '\t' << p.dict.getStr(ab.second) << '\t' << p.pmis[o] / 1024.f << endl;
		}
	};
	if (args.output.empty()) printResult(cout);
	else
	{
		printResult(ofstream{ args.output });
	}
}

void pmiCoherence(const Args& args)
{
	PMIData p = loadPMI(args.model);
	ifstream in{ args.input };
	string line;
	float avgPMI = 0;
	size_t totCnt = 0;
	while (getline(in, line))
	{
		cout << line << endl;
		istringstream str{ line };
		vector<string> words{ istream_iterator<string>{str}, istream_iterator<string>{} };
		size_t cnt = 0, invocab = 0;
		float sumPMI = 0;
		for (size_t i = 0; i < words.size(); ++i)
		{
			if (words[i].empty()) continue;
			invocab++;
			auto a = p.dict.get(words[i]);
			for (size_t j = i + 1; j < words.size(); ++j)
			{
				if (words[j].empty()) continue;
				auto b = p.dict.get(words[i]);
				if (a >= 0 && b >= 0)
				{
					sumPMI += p.pmis[makeTriangleIndex(a, b, p.nVocab)] / 1024.f;
				}
				cnt++;
			}
		}
		cout << "In-vocab Cnt: " << invocab << endl;
		cout << "Average PMI: " << sumPMI / cnt << endl << endl;
		avgPMI += sumPMI / cnt;
		totCnt += 1;
	}

	cout << "========" << endl << "Total Average PMI: " << avgPMI / totCnt << endl;
}

int main(int argc, char* argv[])
{
	Args args;
	try
	{
		cxxopts::Options options("cooc", "Cooccurrence Counter for Multi-core CPU");
		options
			.positional_help("[input field threshold]")
			.show_positional_help();
		auto vpmi = cxxopts::value<string>();
		vpmi->implicit_value("pmi");
		options.add_options()
			("i,input", "Input File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("m,model", "Model File", cxxopts::value<string>(), "Input model file path")
			("o,output", "Output File", cxxopts::value<string>(), "Output file path, default is stdout")
			("f,field", "Field to be counted", cxxopts::value<int>())
			("maxline", "Number of Lines to be read ", cxxopts::value<int>())
			("t,threshold", "Minimum number ", cxxopts::value<int>())
			("h,help", "Help")
			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(thread) for inferencing model, default value is 0 which means the number of cores in system")
			("pmi", "Calculate pointwise mutual informations", vpmi)
			;

		options.parse_positional({ "input", "field", "threshold" });

		try
		{

			auto result = options.parse(argc, argv);

			if (result.count("help") || !result.count("input"))
			{
				cout << options.help({ "" }) << endl;
				return 0;
			}

			if (result.count("input")) args.input = result["input"].as<string>();


#define READ_OPT(P, TYPE) if (result.count(#P)) args.P = result[#P].as<TYPE>()

			READ_OPT(input, string);
			READ_OPT(output, string);
			READ_OPT(model, string);
			READ_OPT(pmi, string);
			READ_OPT(field, int);
			READ_OPT(maxline, int);
			READ_OPT(worker, int);
			READ_OPT(threshold, int);
		}
		catch (const cxxopts::OptionException& e)
		{
			cout << "error parsing options: " << e.what() << endl;
			cout << options.help({ "" }) << endl;
			return -1;
		}

	}
	catch (const cxxopts::OptionException& e)
	{
		cout << "error parsing options: " << e.what() << endl;
		return -1;
	}

	if (args.pmi.empty()) cooc(args);
	else
	{
		if (args.pmi == "pmi") pmi(args);
		else if(args.pmi == "show") pmiShow(args);
		else if (args.pmi == "ch") pmiCoherence(args);
	}
    return 0;
}

