#include "stdafx.h"
#include "CollocationExtractor.h"

using namespace std;

void CollocationExtractor::shrinkDict(size_t cutOff)
{
	WordDictionary tVocab;
	tVocab.add("<UNK>");
	vector<uint32_t> tCnt(1);
	tCnt.reserve(wordCnt.size() / 2);
	for (size_t i = 0; i < wordCnt.size(); ++i)
	{
		if (wordCnt[i] < cutOff) continue;
		tVocab.add(vocab.getStr(i));
		tCnt.emplace_back(wordCnt[i]);
	}
	wordCnt = move(tCnt);
	vocab = move(tVocab);
	wordCnt[0] = totCnt - accumulate(wordCnt.begin(), wordCnt.end(), 0);

}

void CollocationExtractor::updateCohesion()
{
	treeCnt.traverse([this](const vector<uint32_t>& key, pair<uint32_t, float>& value)
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

vector<CollocationExtractor::Collocation> CollocationExtractor::getCollocations(size_t minCnt, float minScore) const
{
	vector<Collocation> ret;
	treeCnt.traverse([this, minCnt, minScore, &ret](const vector<uint32_t>& key, const pair<uint32_t, float>& value)
	{
		if (key.size() <= 1) return vtm_traverse_ret::keep_go;
		if (key.size() >= maxLen) return vtm_traverse_ret::skip_children;
		if (value.first * key.size() < minCnt) return vtm_traverse_ret::skip_children;
		auto be = treeCnt.findChild(key.begin(), key.end());
		float entropy = 0;
		size_t unitUnk = max(5, (int)sqrt(minCnt / 2));
		for (auto it = be.first; it != be.second; ++it)
		{
			auto ch = *it;
			if (ch.first == 0) // for unknown words
			{
				float p = ch.second.first / (float)value.first;
				entropy = -log(p / 5) * p;
			}
			else 
			{
				float p = ch.second.first / (float)value.first;
				entropy = -log(p) * p;
			}
		}
		float score = value.second + log(entropy + 1e-10f);
		if(score < minScore) return vtm_traverse_ret::keep_go;
		vector<const string*> ws(key.size());
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
