#pragma once

#include "utils.h"
#include "vectorTreeMap.hpp"

template<bool SingleChrUnit>
class CollocationExtractor
{
	typedef typename std::conditional<SingleChrUnit, int32_t, std::string>::type unitType;
private:
	WordDictionary<unitType> vocab;
	std::vector<uint32_t> wordCnt;
	vectorTreeMap<uint32_t, std::pair<uint32_t, float>> treeCnt;
	size_t totCnt = 0;
	size_t maxLen;

	std::string _toString(const unitType* ut, std::true_type) const
	{
		if (ut == &vocab.getStr(0)) return "<UNK>";
		return std::wstring_convert<codecvt_utf8<int32_t>, int32_t>{}.to_bytes(*ut);
	}

	std::string _toString(const unitType* ut, std::false_type) const
	{
		if (ut == &vocab.getStr(0)) return "<UNK>";
		return *ut;
	}

public:
	struct Collocation
	{
		std::vector<const unitType*> words;
		uint32_t cnt;
		float logCohesion;
		float entropy;
		float score;

		Collocation(const std::vector<const unitType*>& _words = {}, uint32_t _cnt = 0, float _logCohesion = 0, float _entropy = 0, float _score = 0)
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

	void shrinkDict(size_t cutOff)
	{
		WordDictionary<unitType> tVocab;
		tVocab.add(unitType{});
		std::vector<uint32_t> tCnt(1);
		tCnt.reserve(wordCnt.size() / 2);
		for (size_t i = 0; i < wordCnt.size(); ++i)
		{
			if (wordCnt[i] < cutOff) continue;
			tVocab.add(vocab.getStr(i));
			tCnt.emplace_back(wordCnt[i]);
		}
		wordCnt = std::move(tCnt);
		vocab = std::move(tVocab);
		wordCnt[0] = totCnt - accumulate(wordCnt.begin(), wordCnt.end(), 0);

	}

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

	void updateCohesion()
	{
		treeCnt.traverse([this](const std::vector<uint32_t>& key, std::pair<uint32_t, float>& value)
		{
			if (key.size() <= 1)
			{
				value.second = 0;
				return vtm_traverse_ret::keep_go;
			}
			const auto* parent = treeCnt.find(key.begin(), key.end() - 1);
			float ll = log(value.first / (float)parent->first);
			value.second = (parent->second * (key.size() - 2) + ll) / (key.size() - 1);
			return vtm_traverse_ret::keep_go;
		});
	}

	std::vector<Collocation> getCollocations(size_t minCnt, float minScore) const
	{
		std::vector<Collocation> ret;
		treeCnt.traverse([this, minCnt, minScore, &ret](const std::vector<uint32_t>& key, const std::pair<uint32_t, float>& value)
		{
			if (key.size() <= 1) return vtm_traverse_ret::keep_go;
			if (key.size() >= maxLen) return vtm_traverse_ret::skip_children;
			if (value.first * key.size() < minCnt) return vtm_traverse_ret::skip_children;
			auto be = treeCnt.findChild(key.begin(), key.end());
			float entropy = 0;
			for (auto it = be.first; it != be.second; ++it)
			{
				auto ch = *it;
				if (ch.first == 0) // for unknown words
				{
					float p = ch.second.first / (float)value.first;
					entropy += -log(p / 4) * p;
				}
				else
				{
					float p = ch.second.first / (float)value.first;
					entropy += -log(p) * p;
				}
			}
			float score = value.second + log(entropy + 1e-10f);
			if (score < minScore) return vtm_traverse_ret::keep_go;
			std::vector<const unitType*> ws(key.size());
			transform(key.begin(), key.end(), ws.begin(), [this](uint32_t k)
			{
				return &vocab.getStr(k);
			});
			ret.emplace_back(ws, value.first, value.second, entropy, score);
			return vtm_traverse_ret::keep_go;
		});

		sort(ret.begin(), ret.end(), [](const Collocation& a, const Collocation& b)
		{
			return a.score > b.score;
		});
		return ret;
	}

	const unitType* getUNKWord() const { return &vocab.getStr(0); }

	std::string toString(const unitType* ut) const
	{
		return _toString(ut, std::is_same<unitType, int32_t>{});
	}
};

