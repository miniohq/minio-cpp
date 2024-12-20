// MinIO C++ Library for Amazon S3 Compatible Cloud Storage
// Copyright 2022-2024 MinIO, Inc.
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
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MINIO_CPP_RDMA_H_INCLUDED
#define MINIO_CPP_RDMA_H_INCLUDED

#include "error.h"
#include "client.h"
#include "utils.h"
#include "credentials.h"
#include "signer.h"
#include "rdma-httplib.h"
#include "nvidia-cufile.h"
#include "nvidia-cuobjclient.h"

#define IO_DESC_STR							\
  "0102030405060708:01020304:01020304:0102:010203:1:0102030405060708090a0b0c0d0e0f10:0102030405060708:0102030405060708"

inline constexpr unsigned int kDefaultExpirySeconds =
  (60 * 60 * 24 * 7);  // 7 days

// These functions are invoked by cufile rdma layer either user shadow pages or direct gpu va address
// depending on whether nvidia-fs driver or nv peer mem is present
inline static ssize_t objectPut(const void *handle, const char* buf, size_t size, loff_t offset, const cufileRDMAInfo_t *infop)
{
  void *ctx = cuObjClient::getCtx(handle);
  s3_rdma_client_ctx_t *sctx = static_cast<s3_rdma_client_ctx_t *>(ctx);
  char io_str[sizeof IO_DESC_STR];
  unsigned io_len = sizeof io_str;

  if (infop == nullptr) {
    std::cerr << "obtained NULL descr" << std::endl;
    return -1;
  }

  const std::string descr = std::string(infop->desc_str, infop->desc_len);
  snprintf(io_str, io_len,"%s:%016lx:%016lx;",
	   infop->desc_str, (uint64_t)buf, (uint64_t)size);

  minio::utils::UtcTime date = minio::utils::UtcTime::Now();
  minio::creds::Credentials creds = sctx->provider->Fetch();
  minio::utils::Multimap query_params;  
  minio::http::Url url;
  std::string region = "us-east-1";
  
  if (sctx->uploadId != "") {
    query_params.Add("uploadId", sctx->uploadId);
    if (sctx->partNumber == 0) {
      std::cerr << "partNumber cannot be zero" << std::endl;
      return -1;
    }
    if (sctx->partNumber > 10000) {
      std::cerr << "partNumber cannot be > 10000" << std::endl;
      return -1;
    }
    query_params.Add("partNumber", std::to_string(sctx->partNumber));
  }

  if (minio::error::Error err = sctx->url.BuildUrl(url, minio::http::Method::kPut,
					     region, query_params,
					     sctx->bucket, sctx->object)) {
    std::cerr << "failed to build url. error=" << err
              << ". This should not happen" << std::endl;
    return -1;
  }

  std::string host = url.HostHeaderValue();
  minio::signer::PresignV4(minio::http::Method::kPut,
			   host,
			   url.path,
			   region,
			   query_params,
			   creds.access_key,
			   creds.secret_key,
			   date, kDefaultExpirySeconds);

  std::string path = url.path;
  url.path = "";
  url.query_string = "";
  httplib::Client cli(url.String());

  httplib::Headers headers = {
    {"x-minio-rdma-request", io_str}
  };

  auto res = cli.Put(path+"?"+query_params.ToQueryString(), headers, "", "application/octet-stream");
  if (res.error() != httplib::Error::Success) {
    std::cout << res.error() << std::endl;
    return -1;
  }

  sctx->etag = minio::utils::Trim(res->get_header_value("ETag"), '"');
  return size;
}

inline static ssize_t objectGet(const void *handle, char* buf, size_t size, loff_t offset, const cufileRDMAInfo_t *infop)
{
  void *ctx = cuObjClient::getCtx(handle);
  s3_rdma_client_ctx_t *sctx = static_cast<s3_rdma_client_ctx_t *>(ctx);
  char io_str[sizeof IO_DESC_STR];
  unsigned io_len = sizeof io_str;

  if (infop == nullptr) {
    std::cerr << "obtained NULL descr" << std::endl;
    return -1;
  }

  const std::string descr = std::string(infop->desc_str, infop->desc_len);
  snprintf(io_str, io_len,"%s:%016lx:%016lx;",
	   infop->desc_str, (uint64_t)buf, (uint64_t)size);

  minio::utils::UtcTime date = minio::utils::UtcTime::Now();
  minio::creds::Credentials creds = sctx->provider->Fetch();
  minio::utils::Multimap query_params;  
  minio::http::Url url;
  std::string region = "us-east-1";
  
  if (minio::error::Error err = sctx->url.BuildUrl(url, minio::http::Method::kGet,
					     region, query_params,
					     sctx->bucket, sctx->object)) {
    std::cerr << "failed to build url. error=" << err
              << ". This should not happen" << std::endl;
    return -1;
  }
  
  std::string host = url.HostHeaderValue();
  minio::signer::PresignV4(minio::http::Method::kGet,
			   host,
			   url.path,
			   region,
			   query_params,
			   creds.access_key,
			   creds.secret_key,
			   date, kDefaultExpirySeconds);

  std::string path = url.path;
  url.path = "";
  url.query_string = "";
  httplib::Client cli(url.String());

  httplib::Headers headers = {
    {"x-minio-rdma-request", io_str}
  };

  cli.Get(path+"?"+query_params.ToQueryString(), headers);
  return size;
}

#endif  // _MINIO_CPP_RDMA_H_INCLUDED
