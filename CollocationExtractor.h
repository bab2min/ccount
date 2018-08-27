#pragma once

#include "utils.h"
#include "vectorTreeMap.hpp"

template<bool SingleChrUnit>
class CollocationExtractor
{
	typedef typename std::conditional<SingleChrUnit, int32_t, std::string>::type unitType;
	typedef vectorTreeMap<uint32_t, std::pair<uint32_t, float>> vtmap;
private:
	WordDictionary<unitType> vocab;
	std::vector<uint32_t> wordCnt;
	vtmap forwardCnt;
	vtmap backwardCnt;
	size_t totCnt = 0;
	size_t maxLen;

	std::string _toString(const unitType* ut, std::true_type) const
	{
		if (ut == &vocab.getStr(0)) return "<BEG>";
		if (ut == &vocab.getStr(1)) return "<END>";
		if (ut == &vocab.getStr(2)) return "<UNK>";
		return std::wstring_convert<codecvt_utf8<int32_t>, int32_t>{}.to_bytes(*ut);
	}

	std::string _toString(const unitType* ut, std::false_type) const
	{
		if (ut == &vocab.getStr(0)) return "<BEG>";
		if (ut == &vocab.getStr(1)) return "<END>";
		if (ut == &vocab.getStr(2)) return "<UNK>";
		return *ut;
	}

public:
	struct Collocation
	{
		std::vector<const unitType*> words;
		uint32_t cnt;
		float logCohesion;
		float entropy;
		float backwardLogCohesion;
		float backwardEntropy;
		float score;

		Collocation(const std::vector<const unitType*>& _words = {}, uint32_t _cnt = 0, 
			float _logCohesion = 0, float _entropy = 0, 
			float _backwardLogCohesion = 0, float _backwardEntropy = 0,
			float _score = 0)
			: words(_words), cnt(_cnt), 
			logCohesion(_logCohesion), entropy(_entropy),
			backwardLogCohesion(_backwardLogCohesion), backwardEntropy(_backwardEntropy),
			score(_score)
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
		tVocab.add(unitType{});
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
		ids.emplace_back(0);
		for (auto it = wBegin; it != wEnd; ++it)
		{
			ids.emplace_back(std::max(vocab.get(*it), 2));
		}
		ids.emplace_back(1);

		for (size_t i = 0; i < ids.size(); ++i)
		{
			for (size_t j = i + 1; j < std::min(i + 1 + maxLen, ids.size() + 1); ++j)
			{
				forwardCnt.at(ids.begin() + i, ids.begin() + j).first++;
				backwardCnt.at(ids.rbegin() + i, ids.rbegin() + j).first++;
			}
		}
	}

	void updateCohesion()
	{
		forwardCnt.traverse([this](const std::vector<uint32_t>& key, std::pair<uint32_t, float>& value)
		{
			if (key.size() <= 1)
			{
				value.second = 0;
				return vtm_traverse_ret::keep_go;
			}
			const auto* parent = forwardCnt.find(key.begin(), key.end() - 1);
			float ll = log(value.first / (float)parent->first);
			value.second = (parent->second * (key.size() - 2) + ll) / (key.size() - 1);
			return vtm_traverse_ret::keep_go;
		});

		backwardCnt.traverse([this](const std::vector<uint32_t>& key, std::pair<uint32_t, float>& value)
		{
			if (key.size() <= 1)
			{
				value.second = 0;
				return vtm_traverse_ret::keep_go;
			}
			const auto* parent = backwardCnt.find(key.begin(), key.end() - 1);
			float ll = log(value.first / (float)parent->first);
			value.second = (parent->second * (key.size() - 2) + ll) / (key.size() - 1);
			return vtm_traverse_ret::keep_go;
		});
	}

	std::vector<Collocation> getCollocations(size_t minCnt, float minScore) const
	{
		std::vector<Collocation> ret;
		const auto& calcEntropy = [](const std::pair<vtmap::child_iterator, vtmap::child_iterator>& be, size_t populate)
		{
			float entropy = 0;
			for (auto it = be.first; it != be.second; ++it)
			{
				auto ch = *it;
				if (ch.first <= 2) // for unknown words or end of sent
				{
					float p = ch.second.first / (float)populate;
					entropy += -log(p / 4) * p;
				}
				else
				{
					float p = ch.second.first / (float)populate;
					entropy += -log(p) * p;
				}
			}
			return entropy;
		};

		forwardCnt.traverse([this, minCnt, minScore, &ret, &calcEntropy](const std::vector<uint32_t>& key, const std::pair<uint32_t, float>& value)
		{
			if (key.size() <= 1) return vtm_traverse_ret::keep_go;
			if (key.size() >= maxLen) return vtm_traverse_ret::skip_children;
			if (value.first * key.size() < minCnt) return vtm_traverse_ret::skip_children;
			
			float backwardCohesion = backwardCnt.find(key.rbegin(), key.rend())->second;
			float entropy = calcEntropy(forwardCnt.findChild(key.begin(), key.end()), value.first);
			float backwardEntropy = calcEntropy(backwardCnt.findChild(key.rbegin(), key.rend()), value.first);
			float score = value.second + backwardCohesion + log(entropy + 1e-10f) + log(backwardEntropy + 1e-10f);
			
			if (score < minScore) return vtm_traverse_ret::keep_go;
			std::vector<const unitType*> ws(key.size());
			transform(key.begin(), key.end(), ws.begin(), [this](uint32_t k)
			{
				return &vocab.getStr(k);
			});
			ret.emplace_back(ws, value.first, value.second, entropy, backwardCohesion, backwardEntropy, score);
			return vtm_traverse_ret::keep_go;
		});

		sort(ret.begin(), ret.end(), [](const Collocation& a, const Collocation& b)
		{
			return a.score > b.score;
		});
		return ret;
	}

	const unitType* getBEGWord() const { return &vocab.getStr(0); }
	const unitType* getENDWord() const { return &vocab.getStr(1); }
	const unitType* getUNKWord() const { return &vocab.getStr(2); }

	std::string toString(const unitType* ut) const
	{
		return _toString(ut, std::is_same<unitType, int32_t>{});
	}
};

