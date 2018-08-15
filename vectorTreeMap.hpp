#pragma once

#include <vector>
#include <map>
#include <functional>

enum class vtm_traverse_ret
{
	keep_go = 0,
	skip_children,
	exit,
};

template<class K = int, class V = int>
class vectorTreeMap
{
public:
	enum { npos = -1 };
	struct child_iterator;
	friend struct child_iterator;
protected:
	struct node
	{
		std::map<K, node*> link;
		size_t id = npos;

		~node()
		{
			for (const auto& p : link) delete p.second;
		}
	};
	std::vector<V> values;
	node entry;

	template<class Func>
	vtm_traverse_ret traverse(const node* cur, std::vector<K>& keys, const Func& func) const
	{
		if (cur->id != npos)
		{
			vtm_traverse_ret ret = func(const_cast<const std::vector<K>&>(keys), values[cur->id]);
			if (ret == vtm_traverse_ret::skip_children) return vtm_traverse_ret::keep_go;
			if (ret == vtm_traverse_ret::exit) return vtm_traverse_ret::exit;
		}
		for (const auto& p : cur->link)
		{
			keys.emplace_back(p.first);
			vtm_traverse_ret ret = traverse(p.second, keys, func);
			if (ret == vtm_traverse_ret::exit) return vtm_traverse_ret::exit;
			keys.pop_back();
		}
		return vtm_traverse_ret::keep_go;
	}

	template<class Func>
	vtm_traverse_ret traverse(const node* cur, std::vector<K>& keys, const Func& func)
	{
		if (cur->id != npos)
		{
			vtm_traverse_ret ret = func(const_cast<const std::vector<K>&>(keys), values[cur->id]);
			if (ret == vtm_traverse_ret::skip_children) return vtm_traverse_ret::keep_go;
			if (ret == vtm_traverse_ret::exit) return vtm_traverse_ret::exit;
		}
		for (const auto& p : cur->link)
		{
			keys.emplace_back(p.first);
			vtm_traverse_ret ret = traverse(p.second, keys, func);
			if (ret == vtm_traverse_ret::exit) return vtm_traverse_ret::exit;
			keys.pop_back();
		}
		return vtm_traverse_ret::keep_go;
	}

public:
	struct child_iterator
	{
	protected:
		typename std::map<K, node*>::const_iterator it;
		typename std::map<K, node*>::const_iterator end;
		const vectorTreeMap* cont;
	public:
		child_iterator(const typename std::map<K, node*>::const_iterator& _it = {},
			const typename std::map<K, node*>::const_iterator& _end = {},
			const vectorTreeMap* _cont = nullptr) : it(_it), end(_end), cont(_cont)
		{
			while (it != end && it->second->id == npos) ++it;
		}

		std::pair<K, V> operator*() const
		{
			return std::make_pair(it->first, cont->values[it->second->id]);
		}

		child_iterator& operator++()
		{
			++it;
			while (it != end && it->second->id == npos) ++it;
			return *this;
		}

		bool operator==(const child_iterator& o) const
		{
			return it == o.it;
		}

		bool operator!=(const child_iterator& o) const { return !operator==(o); }
	};


	template<class KIter>
	const V* find(KIter begin, KIter end) const
	{
		const node* cur = &entry;
		for (; begin != end; ++begin)
		{
			auto it = cur->link.find(*begin);
			if (it == cur->link.end()) return nullptr;
			cur = it->second;
		}
		if (cur->id == npos) return nullptr;
		return &values[cur->id];
	}

	template<class KIter>
	V& at(KIter begin, KIter end)
	{
		node* cur = &entry;
		for (; begin != end; ++begin)
		{
			auto it = cur->link.emplace(*begin, nullptr);
			if (it.first->second == nullptr) cur = it.first->second = new node;
			else cur = it.first->second;
		}
		if (cur->id == npos)
		{
			cur->id = values.size();
			values.emplace_back(V{});
		}
		return values[cur->id];
	}

	template<class KIter>
	std::pair<child_iterator, child_iterator> findChild(KIter begin, KIter end) const
	{
		const node* cur = &entry;
		for (; begin != end; ++begin)
		{
			auto it = cur->link.find(*begin);
			if (it == cur->link.end()) return {};
			cur = it->second;
		}
		return std::make_pair(child_iterator{ cur->link.begin(), cur->link.end(), this },
			child_iterator{ cur->link.end(), cur->link.end(), this });
	}

	template<class Func>
	vtm_traverse_ret traverse(const Func& func) const
	{
		std::vector<K> keys;
		return traverse(&entry, keys, func);
	}

	template<class Func>
	vtm_traverse_ret traverse(const Func& func)
	{
		std::vector<K> keys;
		return traverse(&entry, keys, func);
	}
};
