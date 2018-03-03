// CooccurrenceCounter.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ThreadPool.hpp"
#include "cxxopts.hpp"

using namespace std;

class WordDictionary
{
protected:
	unordered_map<string, int> word2id;
	vector<string> id2word;
	mutex mtx;
public:
	enum { npos = (size_t)-1 };
	int add(const string& str)
	{
		if(word2id.emplace(str, word2id.size()).second) id2word.emplace_back(str);
		return word2id.size() - 1;
	}

	int getOrAdd(const string& str)
	{
		lock_guard<mutex> lg(mtx);
		auto it = word2id.find(str);
		if (it != word2id.end()) return it->second;
		return add(str);
	}

	template<class Iter>
	vector<int> getOrAdds(Iter begin, Iter end)
	{
		lock_guard<mutex> lg(mtx);
		vector<int> ret;
		for (; begin != end; ++begin)
		{
			auto it = word2id.find(*begin);
			if (it != word2id.end()) ret.emplace_back(it->second);
			ret.emplace_back(add(*begin));
		}
		return ret;
	}

	int get(const string& str) const
	{
		auto it = word2id.find(str);
		if (it != word2id.end()) return it->second;
		return npos;
	}
	
	string getStr(int id) const
	{
		return id2word[id];
	}
};

void countCooc(vector<uint32_t>& counter, size_t nVocab, const vector<int>& ids)
{
	for (size_t i = 0; i < ids.size(); i++) for (size_t j = i + 1; j < ids.size(); j++)
	{
		auto a = ids[i], b = ids[j];
		if (a == b) continue;
		if (a > b) swap(a, b);
		++counter[a * nVocab + b];
	}
}

template<class LocalData>
vector<LocalData> scanText(istream& input, size_t worker, size_t maxLine, const function<void(LocalData&, string, size_t)>& func, const LocalData& ldInitVal = {})
{
	ThreadPool pool(worker);
	vector<LocalData> ld(worker, ldInitVal);
	string doc;
	int numLine = 0;
	vector<future<void>> futures(worker);
	while (getline(input, doc))
	{
		futures[numLine % futures.size()] = pool.enqueue([&ld, doc, &func](size_t tId, size_t nLine)
		{
			if (tId == 0) cerr << "Line " << nLine << endl;
			return func(ld[tId], doc, nLine);
		}, numLine + 1);
		numLine++;
		if (numLine >= maxLine) break;
	}
	for (auto && p : futures)
	{
		if (p.valid()) p.wait();
	}
	return ld;
}

string selectField(string line, size_t field)
{
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

struct Args
{
	string input, output;
	int field = 0;
	size_t maxline = -1;
	int worker = thread::hardware_concurrency();
	size_t threshold = 100;
};

template<class Type> Type getIS(istream& is)
{
	Type t;
	is >> t;
	return t;
}

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
			unordered_set<string> uniqWords;
			for (string w = getIS<string>(ss); !ss.eof(); ss >> w)
			{
				uniqWords.emplace(w);
			}
			for(auto&& wId : dict.getOrAdds(uniqWords.begin(), uniqWords.end())) ++ld[wId];
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
		unordered_set<int> uniq;
		vector<int> ids;
		for (string w = getIS<string>(ss); !ss.eof(); ss >> w)
		{
			int wId = rdict.get(w);
			if (wId < 0) continue; 
			if (uniq.count(wId)) continue;
			uniq.emplace(wId);
			ids.emplace_back(wId);
		}
		countCooc(ld, nVocab, ids);
	}, vector<uint32_t>(nVocab * nVocab));

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
				for (size_t n = 0; n < nVocab * nVocab; ++n)
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
	for (size_t n = 0; n < nVocab * nVocab; ++n)
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
			out << rdict.getStr(p.first / nVocab) << '\t' << rdict.getStr(p.first % nVocab) << '\t' << p.second << endl;
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

int main(int argc, char* argv[])
{
	Args args;
	try
	{
		cxxopts::Options options("cooc", "Cooccurrence Counter for Multi-core CPU");
		options
			.positional_help("[input field threshold]")
			.show_positional_help();

		options.add_options()
			("i,input", "Input File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("o,output", "Output File", cxxopts::value<string>(), "Input file path that contains documents per line")
			("f,field", "Field to be counted", cxxopts::value<int>())
			("maxline", "Number of Lines to be read ", cxxopts::value<int>())
			("threshold", "Minimum number ", cxxopts::value<int>())
			("h,help", "Help")
			("w,worker", "Number of Workes", cxxopts::value<int>(), "The number of workers(thread) for inferencing model, default value is 0 which means the number of cores in system")
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

	cooc(args);
    return 0;
}

