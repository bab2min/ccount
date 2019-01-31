#include "stdafx.h"
#include "KWordDetector.h"
#include "utils.h"

using namespace std;

void KWordDetector::countUnigram(Counter& cdata, const function<string(size_t)>& reader) const
{
	auto ldUnigram = readProc<unordered_map<string, uint32_t>>(reader, [this, &cdata](string ustr, size_t id, unordered_map<string, uint32_t>& ld)
	{
		stringstream ss{ ustr };
		istream_iterator<string> begin{ ss }, end{};
		for (; begin != end; ++begin)
		{
			++ld[*begin];
		}
	});

	auto& unigramMerged = ldUnigram[0];
	for (size_t i = 1; i < ldUnigram.size(); ++i)
	{
		for (auto& p : ldUnigram[i]) unigramMerged[p.first] += p.second;
	}
	cerr << "Total Vocab Size: " << unigramMerged.size() << endl;

	WordDictionary<string, uint32_t> chrDictShrink;
	chrDictShrink.add("<UNK>");
	chrDictShrink.add("<BEG>");
	chrDictShrink.add("<END>");
	cdata.cntUnigram.resize(3);
	for (auto& p : unigramMerged)
	{
		if (p.second < minCnt) continue;
		chrDictShrink.add(p.first);
		cdata.cntUnigram.emplace_back(p.second);
	}
	cerr << "Selected Vocab Size: " << cdata.cntUnigram.size() << endl;
	cdata.chrDict = move(chrDictShrink);
	cdata.totNum = accumulate(cdata.cntUnigram.begin(), cdata.cntUnigram.end(), 0);
}

void KWordDetector::countBigram(Counter& cdata, const function<string(size_t)>& reader) const
{
	auto ldBigram = readProc<unordered_map<pair<uint16_t, uint16_t>, uint32_t>>(reader, [this, &cdata](string ustr, size_t , unordered_map<pair<uint16_t, uint16_t>, uint32_t>& ld)
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

void atomicIncrease(map<u16light, atomic<uint32_t>>& m, u16light&& k, mutex& insertMtx)
{
	auto it = m.find(k);
	if (it != m.end())
	{
		it->second++;
	}
	else
	{
		lock_guard<mutex> l{ insertMtx };
		++m[k];
	}
}

void KWordDetector::countNgram(Counter& cdata, const function<string(size_t)>& reader) const
{
	mutex forwardInsertMtx, backwardInsertMtx;
	auto ldNgram = readProc<size_t>(reader, [&](string ustr, size_t, size_t ld) 
	{
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

		for (size_t i = 1; i < ids->size(); ++i)
		{
			if (!(*ids)[i]) continue;
			for (size_t j = i + 2; j < min(i + 1 + maxWordLen, ids->size() + 1); ++j)
			{
				if (!(*ids)[j - 1]) break;
				atomicIncrease(cdata.forwardAtmCnt, { ids->begin() + i, ids->begin() + j }, forwardInsertMtx);
				if (!cdata.candBigram.count(make_pair((*ids)[j - 2], (*ids)[j - 1]))) break;
			}
		}

		for (size_t i = 1; i < ids->size(); ++i)
		{
			if (!ids->rbegin()[i]) continue;
			for (size_t j = i + 2; j < min(i + 1 + maxWordLen, ids->size() + 1); ++j)
			{
				if (!ids->rbegin()[j - 1]) break;
				atomicIncrease(cdata.backwardAtmCnt, { ids->rbegin() + i, ids->rbegin() + j }, backwardInsertMtx);
				if (!cdata.candBigram.count(make_pair(ids->rbegin()[j - 1], ids->rbegin()[j - 2]))) break;
			}
		}
	});

	cdata.forwardCnt.insert(cdata.forwardAtmCnt.begin(), cdata.forwardAtmCnt.end());
	cdata.forwardAtmCnt.clear();
	cdata.backwardCnt.insert(cdata.backwardAtmCnt.begin(), cdata.backwardAtmCnt.end());
	cdata.backwardAtmCnt.clear();

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
		if (p.first.size() >= maxWordLen) continue;
		if (p.second < minCnt) continue;
		auto bit = cdata.backwardCnt.find({ p.first.rbegin(), p.first.rend() });
		if (bit == cdata.backwardCnt.end()) continue;

		float forwardCohesion = pow(p.second / (float)cdata.cntUnigram[p.first.front()], 1 / (p.first.size() - 1.f));
		float backwardCohesion = pow(p.second / (float)cdata.cntUnigram[p.first.back()], 1 / (p.first.size() - 1.f));

		float forwardBranch = branchingEntropy(cdata.forwardCnt, it);
		float backwardBranch = branchingEntropy(cdata.backwardCnt, bit);

		float score;

		if (npmiScore) // npmi
		{
			score = p.second;
			for (auto c : p.first) score /= cdata.cntUnigram[c];
			score = log(score) + (p.first.size() - 1) * log(cdata.totNum);
			score /= log(cdata.totNum / (float)p.second) * (p.first.size() - 1);
		}
		else // cohesion
		{
			score = forwardCohesion * backwardCohesion;
			score *= forwardBranch * backwardBranch;
		}
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
