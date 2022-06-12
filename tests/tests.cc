// MinIO C++ Library for Amazon S3 Compatible Cloud Storage
// Copyright 2022 MinIO, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <random>

#include "client.h"

thread_local static std::mt19937 rg{std::random_device{}()};

const static std::string charset =
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

thread_local static std::uniform_int_distribution<std::string::size_type> pick(
    0, charset.length() - 2);

class RandomBuf : public std::streambuf {
 private:
  size_t size_;
  std::array<char, 64> buf_;

 protected:
  int_type underflow() override {
    if (size_ == 0) return EOF;

    size_t size = std::min(size_, buf_.size());
    setg(&buf_[0], &buf_[0], &buf_[size]);
    for (size_t i = 0; i < size; ++i) buf_[i] = charset[pick(rg)];
    size_ -= size;
    return 0;
  }

 public:
  RandomBuf(size_t size) : size_(size) {}
};

class RandCharStream : public std::istream {
 private:
  RandomBuf buf_;

 public:
  RandCharStream(size_t size) : buf_(size) { rdbuf(&buf_); }
};

std::string RandomString(std::string chrs, std::string::size_type length) {
  thread_local static std::uniform_int_distribution<std::string::size_type>
      pick(0, chrs.length() - 2);

  std::string s;
  s.reserve(length);
  while (length--) s += chrs[pick(rg)];
  return s;
}

std::string RandBucketName() {
  return RandomString("0123456789abcdefghijklmnopqrstuvwxyz", 8);
}

std::string RandObjectName() { return RandomString(charset, 8); }

struct MakeBucketError : public std::runtime_error {
  MakeBucketError(std::string err) : runtime_error(err) {}
};

struct RemoveBucketError : public std::runtime_error {
  RemoveBucketError(std::string err) : runtime_error(err) {}
};

struct BucketExistsError : public std::runtime_error {
  BucketExistsError(std::string err) : runtime_error(err) {}
};

class Tests {
 private:
  minio::s3::Client& client_;
  std::string bucket_name_;

 public:
  Tests(minio::s3::Client& client) : client_(client) {
    bucket_name_ = RandBucketName();
    minio::s3::MakeBucketArgs args;
    args.bucket = bucket_name_;
    minio::s3::MakeBucketResponse resp = client_.MakeBucket(args);
    if (!resp) {
      throw std::runtime_error("MakeBucket(): " + resp.Error().String());
    }
  }

  ~Tests() noexcept(false) {
    minio::s3::RemoveBucketArgs args;
    args.bucket = bucket_name_;
    minio::s3::RemoveBucketResponse resp = client_.RemoveBucket(args);
    if (!resp) {
      throw std::runtime_error("RemoveBucket(): " + resp.Error().String());
    }
  }

  void MakeBucket(std::string bucket_name) noexcept(false) {
    minio::s3::MakeBucketArgs args;
    args.bucket = bucket_name;
    minio::s3::MakeBucketResponse resp = client_.MakeBucket(args);
    if (resp) return;
    throw MakeBucketError("MakeBucket(): " + resp.Error().String());
  }

  void RemoveBucket(std::string bucket_name) noexcept(false) {
    minio::s3::RemoveBucketArgs args;
    args.bucket = bucket_name;
    minio::s3::RemoveBucketResponse resp = client_.RemoveBucket(args);
    if (resp) return;
    throw RemoveBucketError("RemoveBucket(): " + resp.Error().String());
  }

  void RemoveObject(std::string bucket_name, std::string object_name) {
    minio::s3::RemoveObjectArgs args;
    args.bucket = bucket_name;
    args.object = object_name;
    minio::s3::RemoveObjectResponse resp = client_.RemoveObject(args);
    if (!resp) {
      throw std::runtime_error("RemoveObject(): " + resp.Error().String());
    }
  }

  void MakeBucket() {
    std::cout << "MakeBucket()" << std::endl;

    std::string bucket_name = RandBucketName();
    MakeBucket(bucket_name);
    RemoveBucket(bucket_name);
  }

  void RemoveBucket() {
    std::cout << "RemoveBucket()" << std::endl;

    std::string bucket_name = RandBucketName();
    MakeBucket(bucket_name);
    RemoveBucket(bucket_name);
  }

  void BucketExists() {
    std::cout << "BucketExists()" << std::endl;

    std::string bucket_name = RandBucketName();
    try {
      MakeBucket(bucket_name);
      minio::s3::BucketExistsArgs args;
      args.bucket = bucket_name;
      minio::s3::BucketExistsResponse resp = client_.BucketExists(args);
      if (!resp) {
        throw BucketExistsError("BucketExists(): " + resp.Error().String());
      }
      if (!resp.exist) {
        throw std::runtime_error("BucketExists(): expected: true; got: false");
      }
      RemoveBucket(bucket_name);
    } catch (const MakeBucketError& err) {
      throw err;
    } catch (const std::runtime_error& err) {
      RemoveBucket(bucket_name);
      throw err;
    }
  }

  void ListBuckets() {
    std::cout << "ListBuckets()" << std::endl;

    std::list<std::string> bucket_names;
    try {
      for (int i = 0; i < 3; i++) {
        std::string bucket_name = RandBucketName();
        MakeBucket(bucket_name);
        bucket_names.push_back(bucket_name);
      }

      minio::s3::ListBucketsResponse resp = client_.ListBuckets();
      if (!resp) {
        throw std::runtime_error("ListBuckets(): " + resp.Error().String());
      }

      int c = 0;
      for (auto& bucket : resp.buckets) {
        if (std::find(bucket_names.begin(), bucket_names.end(), bucket.name) !=
            bucket_names.end()) {
          c++;
        }
      }
      if (c != bucket_names.size()) {
        throw std::runtime_error(
            "ListBuckets(): expected: " + std::to_string(bucket_names.size()) +
            "; got: " + std::to_string(c));
      }
      for (auto& bucket_name : bucket_names) RemoveBucket(bucket_name);
    } catch (const std::runtime_error& err) {
      for (auto& bucket_name : bucket_names) RemoveBucket(bucket_name);
      throw err;
    }
  }

  void StatObject() {
    std::cout << "StatObject()" << std::endl;

    std::string object_name = RandObjectName();

    std::string data = "StatObject()";
    std::stringstream ss(data);
    minio::s3::PutObjectArgs args(ss, data.length(), 0);
    args.bucket = bucket_name_;
    args.object = object_name;
    minio::s3::PutObjectResponse resp = client_.PutObject(args);
    if (!resp) {
      throw std::runtime_error("PutObject(): " + resp.Error().String());
    }
    try {
      minio::s3::StatObjectArgs args;
      args.bucket = bucket_name_;
      args.object = object_name;
      minio::s3::StatObjectResponse resp = client_.StatObject(args);
      if (!resp) {
        throw std::runtime_error("StatObject(): " + resp.Error().String());
      }
      if (resp.size != data.length()) {
        throw std::runtime_error(
            "StatObject(): expected: " + std::to_string(data.length()) +
            "; got: " + std::to_string(resp.size));
      }
      RemoveObject(bucket_name_, object_name);
    } catch (const std::runtime_error& err) {
      RemoveObject(bucket_name_, object_name);
      throw err;
    }
  }

  void RemoveObject() {
    std::cout << "RemoveObject()" << std::endl;

    std::string object_name = RandObjectName();
    std::string data = "RemoveObject()";
    std::stringstream ss(data);
    minio::s3::PutObjectArgs args(ss, data.length(), 0);
    args.bucket = bucket_name_;
    args.object = object_name;
    minio::s3::PutObjectResponse resp = client_.PutObject(args);
    if (!resp) {
      throw std::runtime_error("PutObject(): " + resp.Error().String());
    }
    RemoveObject(bucket_name_, object_name);
  }

  void DownloadObject() {
    std::cout << "DownloadObject()" << std::endl;

    std::string object_name = RandObjectName();

    std::string data = "DownloadObject()";
    std::stringstream ss(data);
    minio::s3::PutObjectArgs args(ss, data.length(), 0);
    args.bucket = bucket_name_;
    args.object = object_name;
    minio::s3::PutObjectResponse resp = client_.PutObject(args);
    if (!resp) {
      throw std::runtime_error("PutObject(): " + resp.Error().String());
    }

    try {
      std::string filename = RandObjectName();
      minio::s3::DownloadObjectArgs args;
      args.bucket = bucket_name_;
      args.object = object_name;
      args.filename = filename;
      minio::s3::DownloadObjectResponse resp = client_.DownloadObject(args);
      if (!resp) {
        throw std::runtime_error("DownloadObject(): " + resp.Error().String());
      }

      std::ifstream file(filename);
      file.seekg(0, std::ios::end);
      size_t length = file.tellg();
      file.seekg(0, std::ios::beg);
      char* buf = new char[length];
      file.read(buf, length);
      file.close();

      if (data != std::string(buf, length)) {
        throw std::runtime_error("DownloadObject(): expected: " + data +
                                 "; got: " + buf);
      }
      std::filesystem::remove(filename);
      RemoveObject(bucket_name_, object_name);
    } catch (const std::runtime_error& err) {
      RemoveObject(bucket_name_, object_name);
      throw err;
    }
  }

  void GetObject() {
    std::cout << "GetObject()" << std::endl;

    std::string object_name = RandObjectName();

    std::string data = "GetObject()";
    std::stringstream ss(data);
    minio::s3::PutObjectArgs args(ss, data.length(), 0);
    args.bucket = bucket_name_;
    args.object = object_name;
    minio::s3::PutObjectResponse resp = client_.PutObject(args);
    if (!resp) {
      throw std::runtime_error("PutObject(): " + resp.Error().String());
    }

    try {
      minio::s3::GetObjectArgs args;
      args.bucket = bucket_name_;
      args.object = object_name;
      std::string content;
      args.datafunc =
          [&content = content](minio::http::DataFunctionArgs args) -> bool {
        content += args.datachunk;
        return true;
      };
      minio::s3::GetObjectResponse resp = client_.GetObject(args);
      if (!resp) {
        throw std::runtime_error("GetObject(): " + resp.Error().String());
      }
      if (data != content) {
        throw std::runtime_error("GetObject(): expected: " + data +
                                 "; got: " + content);
      }
      RemoveObject(bucket_name_, object_name);
    } catch (const std::runtime_error& err) {
      RemoveObject(bucket_name_, object_name);
      throw err;
    }
  }

  void ListObjects() {
    std::cout << "ListObjects()" << std::endl;

    std::list<std::string> object_names;
    try {
      for (int i = 0; i < 3; i++) {
        std::string object_name = RandObjectName();
        std::stringstream ss;
        minio::s3::PutObjectArgs args(ss, 0, 0);
        args.bucket = bucket_name_;
        args.object = object_name;
        minio::s3::PutObjectResponse resp = client_.PutObject(args);
        if (!resp) {
          throw std::runtime_error("PutObject(): " + resp.Error().String());
        }
        object_names.push_back(object_name);
      }

      int c = 0;
      minio::s3::ListObjectsArgs args;
      args.bucket = bucket_name_;
      minio::s3::ListObjectsResult result = client_.ListObjects(args);
      for (; result; result++) {
        minio::s3::Item item = *result;
        if (!item) {
          throw std::runtime_error("ListObjects(): " + item.Error().String());
        }
        if (std::find(object_names.begin(), object_names.end(), item.name) !=
            object_names.end()) {
          c++;
        }
      }

      if (c != object_names.size()) {
        throw std::runtime_error(
            "ListObjects(): expected: " + std::to_string(object_names.size()) +
            "; got: " + std::to_string(c));
      }
      for (auto& object_name : object_names) {
        RemoveObject(bucket_name_, object_name);
      }
    } catch (const std::runtime_error& err) {
      for (auto& object_name : object_names) {
        RemoveObject(bucket_name_, object_name);
      }
      throw err;
    }
  }

  void PutObject() {
    std::cout << "PutObject()" << std::endl;

    {
      std::string object_name = RandObjectName();
      std::string data = "PutObject()";
      std::stringstream ss(data);
      minio::s3::PutObjectArgs args(ss, data.length(), 0);
      args.bucket = bucket_name_;
      args.object = object_name;
      minio::s3::PutObjectResponse resp = client_.PutObject(args);
      if (!resp) {
        throw std::runtime_error("PutObject(): " + resp.Error().String());
      }
      RemoveObject(bucket_name_, object_name);
    }

    {
      std::string object_name = RandObjectName();
      size_t size = 13930573;
      RandCharStream stream(size);
      minio::s3::PutObjectArgs args(stream, size, 0);
      args.bucket = bucket_name_;
      args.object = object_name;
      minio::s3::PutObjectResponse resp = client_.PutObject(args);
      if (!resp) {
        throw std::runtime_error("<Multipart> PutObject(): " +
                                 resp.Error().String());
      }
      RemoveObject(bucket_name_, object_name);
    }
  }

  void CopyObject() {
    std::cout << "CopyObject()" << std::endl;

    std::string object_name = RandObjectName();
    std::string src_object_name = RandObjectName();
    std::string data = "CopyObject()";
    std::stringstream ss(data);
    minio::s3::PutObjectArgs args(ss, data.length(), 0);
    args.bucket = bucket_name_;
    args.object = src_object_name;
    minio::s3::PutObjectResponse resp = client_.PutObject(args);
    if (!resp) {
      throw std::runtime_error("PutObject(): " + resp.Error().String());
    }

    try {
      minio::s3::CopySource source;
      source.bucket = bucket_name_;
      source.object = src_object_name;
      minio::s3::CopyObjectArgs args;
      args.bucket = bucket_name_;
      args.object = object_name;
      args.source = source;
      minio::s3::CopyObjectResponse resp = client_.CopyObject(args);
      if (!resp) {
        throw std::runtime_error("CopyObject(): " + resp.Error().String());
      }
      RemoveObject(bucket_name_, src_object_name);
      RemoveObject(bucket_name_, object_name);
    } catch (const std::runtime_error& err) {
      RemoveObject(bucket_name_, src_object_name);
      RemoveObject(bucket_name_, object_name);
      throw err;
    }
  }

  void UploadObject() {
    std::cout << "UploadObject()" << std::endl;

    std::string data = "UploadObject()";
    std::string filename = RandObjectName();
    std::ofstream file(filename);
    file << data;
    file.close();

    std::string object_name = RandObjectName();
    minio::s3::UploadObjectArgs args;
    args.bucket = bucket_name_;
    args.object = object_name;
    args.filename = filename;
    minio::s3::UploadObjectResponse resp = client_.UploadObject(args);
    if (!resp) {
      throw std::runtime_error("UploadObject(): " + resp.Error().String());
    }
    std::filesystem::remove(filename);
    RemoveObject(bucket_name_, object_name);
  }
};  // class Tests

int main(int argc, char* argv[]) {
  std::string host;
  if (!minio::utils::GetEnv(host, "S3HOST")) {
    std::cerr << "S3HOST environment variable must be set" << std::endl;
    return EXIT_FAILURE;
  }

  std::string access_key;
  if (!minio::utils::GetEnv(access_key, "ACCESS_KEY")) {
    std::cerr << "ACCESS_KEY environment variable must be set" << std::endl;
    return EXIT_FAILURE;
  }

  std::string secret_key;
  if (!minio::utils::GetEnv(secret_key, "SECRET_KEY")) {
    std::cerr << "SECRET_KEY environment variable must be set" << std::endl;
    return EXIT_FAILURE;
  }

  std::string value;
  bool secure = true;
  if (minio::utils::GetEnv(value, "IS_HTTP")) secure = false;

  bool ignore_cert_check = false;
  if (minio::utils::GetEnv(value, "IGNORE_CERT_CHECK")) secure = true;

  std::string region;
  minio::utils::GetEnv(region, "REGION");

  minio::s3::BaseUrl base_url(host, secure);

  minio::creds::StaticProvider provider(access_key, secret_key);
  minio::s3::Client client(base_url, &provider);

  Tests tests(client);
  tests.MakeBucket();
  tests.RemoveBucket();
  tests.BucketExists();
  tests.ListBuckets();
  tests.StatObject();
  tests.RemoveObject();
  tests.DownloadObject();
  tests.GetObject();
  tests.ListObjects();
  tests.PutObject();
  tests.CopyObject();
  tests.UploadObject();

  return EXIT_SUCCESS;
}