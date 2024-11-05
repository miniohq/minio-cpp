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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include <miniocpp/client.h>

int main(int argc, char* argv[]) {
  std::string host;
  std::string access_key;
  std::string secret_key;

  char *bufptr;
  size_t bufsize = 101 * 1024 * 1024UL;

  if (argc <= 1) {
    printf("usage: %s <server_address>\n", argv[0]);
    exit(1);
  }

  if (argc > 1) {
    host = std::string(argv[1]);
    access_key = std::string(argv[2]);
    secret_key = std::string(argv[3]);
    if (argc == 5) {
      bufsize = std::atoi(argv[4]);
    }
  }

  // Create S3 base URL.
  minio::s3::BaseUrl base_url(host, false, "us-east-1");

  // Create credential provider.
  minio::creds::StaticProvider provider(access_key, secret_key);

  // Create S3 client.
  minio::s3::Client client(base_url, &provider);

  int res = posix_memalign((void **)&bufptr, getpagesize(), bufsize);
  if (res) {
    std::cerr << "unable to allocate system memory with alignment"
	      << getpagesize() << "buf size"
	      << bufsize << std::endl;
  }
  assert(bufptr);

  // Create get object arguments.
  minio::s3::GetObjectRDMAArgs args(bufptr, bufsize);
  args.bucket = "my-bucket";
  args.object = "my-object";

  // Call get object.
  minio::s3::GetObjectResponse resp = client.GetObject(args);

  // Handle response.
  if (resp) {
    std::cout << std::endl
              << "data of my-object is received successfully" << std::endl;
  } else {
    std::cout << "unable to get object; " << resp.Error().String() << std::endl;
  }

  // Open the file in binary mode for writing
  std::ofstream file("output.txt", std::ios::binary);
  if (file.is_open()) {
    // Write the buffer to the file
    file.write(bufptr, bufsize);
    
    // Close the file
    file.close();
    
    std::cout << "Buffer written to file successfully." << std::endl;
  } else {
    std::cerr << "Error opening file." << std::endl;
  }
  
  free(bufptr);

  return 0;
}
