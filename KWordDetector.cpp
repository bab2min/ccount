#include "stdafx.h"
#include "KWordDetector.h"
#include "Utils.h"

using namespace std;

void KWordDetector::countUnigram(Counter& cdata, const function<string(size_t)>& reader) const
{
	auto ldUnigram = readProc<vector<uint32_t>>(reader, [this, &cdata](string ustr, size_t id, vector<uint32_t>& ld)
	{
		stringstream ss{ ustr };
		istream_iterator<string> begin{ ss }, end{};
		vector<uint32_t> ids = cdata.chrDict.getOrAdds(begin, end);
		if (ids.empty()) return;
		uint32_t maxId = *max_element(ids.begin(), ids.end());
		if (ld.size() <= maxId) ld.resize(maxId + 1);
		for (auto id : ids) ld[id]++;
	});

	uint32_t maxId = 0;
	for (auto& t : ldUnigram)
	{
		if (maxId < t.size()) maxId = t.size();
	}
	cerr << "Total Vocab Size: " << maxId << endl;
	auto& unigramMerged = ldUnigram[0];
	unigramMerged.resize(maxId);
	for (size_t i = 1; i < ldUnigram.size(); ++i)
	{
		for (size_t n = 0; n < ldUnigram[i].size(); ++n) unigramMerged[n] += ldUnigram[i][n];
	}

	WordDictionary<string, uint32_t> chrDictShrink;
	chrDictShrink.add("<UNK>");
	chrDictShrink.add("<BEG>");
	chrDictShrink.add("<END>");
	cdata.cntUnigram.resize(3);
	for (size_t i = 0; i < maxId; ++i)
	{
		if (unigramMerged[i] < minCnt) continue;
		chrDictShrink.add(cdata.chrDict.getStr(i));
		cdata.cntUnigram.emplace_back(unigramMerged[i]);
	}
	cerr << "Selected Vocab Size: " << cdata.cntUnigram.size() << endl;
	cdata.chrDict = move(chrDictShrink);
}

void KWordDetector::countBigram(Counter& cdata, const function<string(size_t)>& reader) const
{
	auto ldBigram = readProc<unordered_map<pair<uint16_t, uint16_t>, uint32_t>>(reader, [this, &cdata](string ustr, size_t id, unordered_map<pair<uint16_t, uint16_t>, uint32_t>& ld)
	{
		stringstream ss{ ustr };
		istream_iterator<string> begin{ ss }, end{};
		uint16_t a = 1;
		for (; begin != end; ++begin)
		{
			uint16_t id = cdata.chrDict.get(*begin);
			if (id == (uint16_t)-1) id = 0;
			if (a && id) ++ld[make_pair(a, id)];
			a = id;
		}
		if (a) ++ld[make_pair(a, 2)];
	});

	auto& bigramMerged = ldBigram[0];
	for (size_t i = 1; i < ldBigram.size(); ++i)
	{
		for (auto& p : ldBigram[i]) bigramMerged[p.first] += p.second;
	}

	for (auto& p : bigramMerged)
	{
		if (p.second < minCnt) continue;
		cdata.candBigram.emplace(p.first);
	}
}

void KWordDetector::countNgram(Counter& cdata, const function<string(size_t)>& reader) const
{
	vector<ReusableThread> rt(2);
	vector<future<void>> futures(2);
	for (size_t id = 0; ; ++id)
	{
		auto ustr = reader(id);
		if (ustr.empty()) break;
		stringstream ss{ ustr };
		istream_iterator<string> begin{ ss }, end{};
		auto ids = make_shared<vector<uint16_t>>();
		ids->emplace_back(1);
		for (; begin != end; ++begin)
		{
			uint16_t id = cdata.chrDict.get(*begin);
			if (id == (uint16_t)-1) id = 0;
			ids->emplace_back(id);
		}
		ids->emplace_back(2);

		if (futures[0].valid()) futures[0].get();
		futures[0] = rt[0].setWork([&, ids]()
		{
			for (size_t i = 1; i < ids->size(); ++i)
			{
				if (!(*ids)[i]) continue;
				for (size_t j = i + 2; j < min(i + 1 + maxWordLen, ids->size() + 1); ++j)
				{
					if (!(*ids)[j - 1]) break;
					++cdata.forwardCnt[{ids->begin() + i, ids->begin() + j}];
					if (!cdata.candBigram.count(make_pair((*ids)[j - 2], (*ids)[j - 1]))) break;
				}
			}
		});
			
		if (futures[1].valid()) futures[1].get();
		futures[1] = rt[1].setWork([&, ids]()
		{
			for (size_t i = 1; i < ids->size(); ++i)
			{
				if (!ids->rbegin()[i]) continue;
				for (size_t j = i + 2; j < min(i + 1 + maxWordLen, ids->size() + 1); ++j)
				{
					if (!ids->rbegin()[j - 1]) break;
					++cdata.backwardCnt[{ids->rbegin() + i, ids->rbegin() + j}];
					if (!cdata.candBigram.count(make_pair(ids->rbegin()[j - 1], ids->rbegin()[j - 2]))) break;
				}
			}
		});
	}

	for (auto& f : futures)
	{
		if (f.valid()) f.get();
	}
	
	u16light prefixToErase = {};
	for (auto it  = cdata.forwardCnt.cbegin(); it != cdata.forwardCnt.cend();)
	{
		auto& p = *it;
		if (prefixToErase.empty() || !p.first.startsWith(prefixToErase))
		{
			if (p.second < minCnt) prefixToErase = p.first;
			else prefixToErase = {};
			++it;
		}
		else
		{
			it = cdata.forwardCnt.erase(it);
		}
	}

	prefixToErase = {};
	for (auto it = cdata.backwardCnt.cbegin(); it != cdata.backwardCnt.cend();)
	{
		auto& p = *it;
		if (prefixToErase.empty() || !p.first.startsWith(prefixToErase))
		{
			if (p.second < minCnt) prefixToErase = p.first;
			else prefixToErase = {};
			++it;
		}
		else
		{
			it = cdata.backwardCnt.erase(it);
		}
	}
}

float KWordDetector::branchingEntropy(const map<u16light, uint32_t>& cnt, map<u16light, uint32_t>::iterator it) const
{
	u16light endKey = it->first;
	float tot = it->second;
	size_t len = endKey.size();
	endKey.back()++;
	++it;
	auto eit = cnt.lower_bound(endKey);
	size_t sum = 0;
	float entropy = 0;
	for (; it != eit; ++it)
	{
		if (it->first.size() != len + 1) continue;
		sum += it->second;
		float p = it->second / tot;
		if (it->first.back() < 3)
		{
			entropy -= p * log(p / 4);
		}
		else
		{
			entropy -= p * log(p);
		}
	}

	if (sum < tot)
	{
		float p = (tot - sum) / tot;
		entropy -= p * log(p / ((tot - sum) / minCnt));
	}

	return entropy;
}

vector<KWordDetector::WordInfo> KWordDetector::extractWords(const std::function<std::string(size_t)>& reader) const
{
	Counter cdata;
	cerr << "Scanning..." << endl;
	countUnigram(cdata, reader);
	countBigram(cdata, reader);
	cerr << "Counting..." << endl;
	countNgram(cdata, reader);

	cerr << "Calculating..." << endl;
	vector<WordInfo> ret;
	for (auto it = cdata.forwardCnt.begin(); it != cdata.forwardCnt.end(); ++it)
	{
		auto& p = *it;
		if (p.second < minCnt) continue;
		auto bit = cdata.backwardCnt.find({ p.first.rbegin(), p.first.rend() });
		if (bit == cdata.backwardCnt.end()) continue;

		float forwardCohesion = pow(p.second / (float)cdata.cntUnigram[p.first.front()], 1 / (p.first.size() - 1.f));
		float backwardCohesion = pow(p.second / (float)cdata.cntUnigram[p.first.back()], 1 / (p.first.size() - 1.f));

		float forwardBranch = branchingEntropy(cdata.forwardCnt, it);
		float backwardBranch = branchingEntropy(cdata.backwardCnt, bit);

		float score = forwardCohesion * backwardCohesion;
		score *= forwardBranch * backwardBranch;
		if (score < minScore) continue;
		vector<string> form;
		form.reserve(p.first.size());
		transform(p.first.begin(), p.first.end(), back_inserter(form), [this, &cdata](char16_t c) { return cdata.chrDict.getStr(c); });

		ret.emplace_back(move(form), score, backwardBranch, forwardBranch, backwardCohesion, forwardCohesion, p.second);
	}

	sort(ret.begin(), ret.end(), [](const auto& a, const auto& b)
	{
		return a.score > b.score;
	});
	return ret;
}
