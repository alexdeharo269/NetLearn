#ifndef CONFIG_H
#define CONFIG_H
// ─────────────────────────────────────────────────────────────────────────────
// Config — a tiny `key = value` reader.
//
// Why this exists: every numeric knob the thesis cares about (tau, ridge, train
// ratio, steps, washout, reservoir size, number of surrogate / Win replicates, …)
// must be editable from the FIRST cell of the notebook WITHOUT recompiling the C++.
// The notebook writes a plain text `config.txt`, the runner reads it here. Lines
// beginning with '#' and blank lines are ignored; everything else is `key = value`.
// ─────────────────────────────────────────────────────────────────────────────
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>

struct Config
{
    std::map<std::string, std::string> kv;

    void load(const std::string &path)
    {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line))
        {
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            trim(k); trim(v);
            if (!k.empty()) kv[k] = v;
        }
    }

    static void trim(std::string &s)
    {
        const char *ws = " \t\r\n";
        size_t a = s.find_first_not_of(ws);
        if (a == std::string::npos) { s.clear(); return; }
        size_t b = s.find_last_not_of(ws);
        s = s.substr(a, b - a + 1);
    }

    std::string gets(const std::string &k, const std::string &d = "") const
    { auto it = kv.find(k); return it == kv.end() ? d : it->second; }

    int geti(const std::string &k, int d) const
    { auto it = kv.find(k); return it == kv.end() ? d : std::stoi(it->second); }

    double getd(const std::string &k, double d) const
    { auto it = kv.find(k); return it == kv.end() ? d : std::stod(it->second); }

    bool getb(const std::string &k, bool d) const
    {
        auto it = kv.find(k);
        if (it == kv.end()) return d;
        const std::string &v = it->second;
        return v == "1" || v == "true" || v == "True" || v == "yes" || v == "on";
    }

    // Parse a comma-separated list of ints, e.g. "76,77".
    std::vector<int> getints(const std::string &k) const
    {
        std::vector<int> out;
        auto it = kv.find(k);
        if (it == kv.end()) return out;
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            trim(tok);
            if (!tok.empty()) out.push_back(std::stoi(tok));
        }
        return out;
    }
};

#endif
