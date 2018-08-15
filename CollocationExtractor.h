#pragma once

#include "utils.h"
#include "vectorTreeMap.hpp"

class CollocationExtractor
{
private:
	WordDictionary vocab;
	std::vector<uint32_t> wordCnt;
	vectorTreeMap<uint32_t, std::pair<uint32_t, float>> treeCnt;
	size_t totCnt = 0;
	size_t maxLen;
public:
	struct Collocation
	{
		std::vector<const std::string*> words;
		uint32_t cnt;
		float logCohesion;
		float entropy;
		float score;

		Collocation(const std::vector<const std::string*>& _words = {}, uint32_t _cnt = 0, float _logCohesion = 0, float _entropy = 0, float _score = 0)
			: words(_words), cnt(_cnt), logCohesion(_logCohesion), entropy(_entropy), score(_score)
		{
		}
	};

	CollocationExtractor(size_t _maxLen = 4) : maxLen(_maxLen) {}

	template<class Iter>
	void countWords(Iter wBegin, Iter wEnd)
	{
		auto ids = vocab.getOrAdds(wBegin, wEnd);
		int max_id = *std::max_element(ids.begin(), ids.end());
		if (wordCnt.size() <= max_id) wordCnt.resize(max_id + 1);
		for (auto i : ids) wordCnt[i]++;
		totCnt += ids.size();
	}

	void shrinkDict(size_t cutOff);

	template<class Iter>
	void countNgrams(Iter wBegin, Iter wEnd)
	{
		std::vector<int> ids;
		for (auto it = wBegin; it != wEnd; ++it)
		{
			ids.emplace_back(std::max(vocab.get(*it), 0));
		}

		for (size_t i = 0; i < ids.size(); ++i)
		{
			for (size_t j = i + 1; j < std::min(i + maxLen + 1, ids.size()); ++j)
			{
				treeCnt.at(ids.begin() + i, ids.begin() + j).first++;
			}
		}
	}

	void updateCohesion();

	std::vector<Collocation> getCollocations(size_t minCnt, float minScore) const;

	const std::string* getUNKWord() const { return &vocab.getStr(0); }
};

