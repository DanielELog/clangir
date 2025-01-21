//===- ProtoEnumsGenHelper.cpp - MLIR enum generator helpers --------------===//
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

#include "ProtoEnumsGenHelper.h"
#include "mlir/TableGen/Format.h"

llvm::StringRef makeIdentifier(llvm::StringRef str) {
  if (!str.empty() && llvm::isDigit(static_cast<unsigned char>(str.front()))) {
    std::string newStr = std::string("_") + str.str();
    return newStr;
  }
  return str.str();
}

llvm::StringRef makeProtoSymbol(llvm::StringRef symbol) {
    return llvm::convertToCamelFromSnakeCase(symbol, true);
}

llvm::StringRef makeFullProtoSymbol(llvm::StringRef enumName,
                                    llvm::StringRef symbol) {
    return llvm::formatv("{0}_{1}", enumName, symbol).str();
}
