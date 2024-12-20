/*
 * SPDX-FileCopyrightText: Copyright (c) 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef _CUOBJCLIENT_H_
#define _CUOBJCLIENT_H_

#define OBJ_RDMA_V1 "CUOBJ"

#include <unistd.h>
#include <stdio.h>
#include "request.h"
#include "nvidia-cufile.h"
#include "providers.h"

typedef enum cuObjOpType_enum {
  CUOBJ_GET = 0,
  CUOBJ_PUT = 1,
  CUOBJ_INVALID=9999
} cuObjOpType_t;

typedef struct s3_rdma_client_ctx {
  minio::creds::Provider* const provider = nullptr;
  std::string bucket;
  std::string object;
  std::string uploadId;
  size_t partNumber;
  std::string etag;
  minio::s3::BaseUrl url;
  cuObjOpType_t op;
} s3_rdma_client_ctx_t;

typedef enum cuObjErr_enum {
  CU_OBJ_SUCCESS =0,
  CU_OBJ_FAIL =1,
} cuObjErr_t;	

typedef enum cuObjProto_enum {
  CUOBJ_PROTO_RDMA_DC_V1=1001,
  CUOBJ_PROTO_MAX
} cuObjProto_t;

typedef struct CUObjIOOps {
  /* NULL means try VFS */
  ssize_t (*get) (const void *ctx, char*, size_t, loff_t, const cufileRDMAInfo_t*);
  ssize_t (*put) (const void *ctx, const char *, size_t, loff_t , const cufileRDMAInfo_t*);
} CUObjOps_t;

class cuObjClient {
public:
  cuObjClient(CUObjOps_t& ops, cuObjProto_t proto=CUOBJ_PROTO_RDMA_DC_V1);
  ~cuObjClient();

  cuObjErr_t cuMemObjGetDescriptor(void *ptr, size_t size);
  cuObjErr_t cuMemObjPutDescriptor(void *ptr);
  static void* getCtx(const void *handle);
  ssize_t cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);
  ssize_t cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);
  bool isConnected(void);
private:
  bool cuObjRegisterKey();
  void* _ctx;
  CUfileHandle_t _cufh;
  CUfileFSOps _objectFsOps;
  bool _connected;
  cuObjProto_t _proto;
};

#endif
