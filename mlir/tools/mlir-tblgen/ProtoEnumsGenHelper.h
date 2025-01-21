//===- ProtoEnumsGenHelper.h - MLIR enum generator helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the enum generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_PROTOENUMSGENHELPER_H_
#define MLIR_TOOLS_MLIRTBLGEN_PROTOENUMSGENHELPER_H_

#include "llvm/ADT/StringExtras.h"

static llvm::StringRef makeIdentifier(llvm::StringRef str);

static llvm::StringRef makeProtoSymbol(llvm::StringRef symbol);

static llvm::StringRef makeFullProtoSymbol(llvm::StringRef enumName,
                                           llvm::StringRef protoSymbol);

#endif // MLIR_TOOLS_MLIRTBLGEN_PROTOENUMSGENHELPER_H_
