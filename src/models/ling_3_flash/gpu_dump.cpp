#include "src/models/ling_3_flash/gpu_dump.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace gufo::models::ling_3_flash {
namespace {

std::string LayerSuffix(int layer) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), ".l%02d", layer);
  return buf;
}

}  // namespace

bool GpuDumpRecorder::Wants(std::string_view name, int layer) const {
  if (!names_.empty() && names_.find(name) == names_.end()) {
    return false;
  }
  return layers_.empty() || layers_.count(layer) > 0;
}

void GpuDumpRecorder::Float(std::string_view name, int layer,
                            std::uint32_t width, std::span<const float> rows) {
  if (!Wants(name, layer)) {
    return;
  }
  Series& s = series_[{std::string(name), layer}];
  s.width = width;
  s.f.insert(s.f.end(), rows.begin(), rows.end());
}

void GpuDumpRecorder::Int(std::string_view name, int layer, std::uint32_t width,
                          std::span<const std::int32_t> rows) {
  if (!Wants(name, layer)) {
    return;
  }
  Series& s = series_[{std::string(name), layer}];
  s.is_int = true;
  s.width = width;
  s.i.insert(s.i.end(), rows.begin(), rows.end());
}

std::span<const float> GpuDumpRecorder::Get(const std::string& name, int layer,
                                            std::uint32_t* width) const {
  const auto it = series_.find({name, layer});
  if (it == series_.end() || it->second.is_int) {
    return {};
  }
  if (width != nullptr) {
    *width = it->second.width;
  }
  return it->second.f;
}

std::span<const std::int32_t> GpuDumpRecorder::GetInt(const std::string& name,
                                                      int layer) const {
  const auto it = series_.find({name, layer});
  if (it == series_.end() || !it->second.is_int) {
    return {};
  }
  return it->second.i;
}

bool GpuDumpRecorder::Write(const std::string& dir,
                            const std::string& extra_entries,
                            const std::string& extra_fields,
                            std::string* error_msg) const {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    if (error_msg != nullptr) {
      *error_msg = "cannot create " + dir + ": " + ec.message();
    }
    return false;
  }
  std::string entries;
  for (const auto& [key, s] : series_) {
    const std::string file = key.first +
                             (key.second >= 0 ? LayerSuffix(key.second) : "") +
                             (s.is_int ? ".i32" : ".f32");
    std::ofstream out(dir + "/" + file, std::ios::binary);
    const std::size_t count = s.is_int ? s.i.size() : s.f.size();
    if (s.is_int) {
      out.write(reinterpret_cast<const char*>(s.i.data()),
                static_cast<std::streamsize>(count * sizeof(std::int32_t)));
    } else {
      out.write(reinterpret_cast<const char*>(s.f.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    }
    if (!out) {
      if (error_msg != nullptr) {
        *error_msg = "write failed: " + dir + "/" + file;
      }
      return false;
    }
    const std::size_t tokens = s.width == 0 ? 0 : count / s.width;
    if (!entries.empty()) {
      entries += ",\n";
    }
    entries += "    {\"name\": \"" + key.first +
               "\", \"layer\": " + std::to_string(key.second) +
               ", \"file\": \"" + file + "\", \"dtype\": \"" +
               (s.is_int ? "i32" : "f32") + "\", \"shape\": [" +
               std::to_string(tokens) + ", " + std::to_string(s.width) +
               "], \"tokens\": " + std::to_string(tokens) + "}";
  }
  if (!extra_entries.empty()) {
    if (!entries.empty()) {
      entries += ",\n";
    }
    entries += extra_entries;
  }
  std::ofstream index(dir + "/index.json");
  index << "{\n  \"format\": \"gufo-ling-dump-v1\",\n"
        << extra_fields << "  \"entries\": [\n"
        << entries << "\n  ]\n}\n";
  if (!index) {
    if (error_msg != nullptr) {
      *error_msg = "write failed: " + dir + "/index.json";
    }
    return false;
  }
  return true;
}

bool ReadF32Dump(const std::string& path, std::vector<float>* out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return false;
  }
  const auto bytes = static_cast<std::size_t>(in.tellg());
  if (bytes % sizeof(float) != 0) {
    return false;
  }
  out->resize(bytes / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(out->data()),
          static_cast<std::streamsize>(bytes));
  return static_cast<bool>(in);
}

}  // namespace gufo::models::ling_3_flash
