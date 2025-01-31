#include "OpGenHelpers.h"
#include "VespaGen.h"

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Successor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

#include <set>

using llvm::formatv;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::Attribute;
using mlir::tblgen::getRequestedOpDefinitions;
using mlir::tblgen::NamedSuccessor;
using mlir::tblgen::Operator;

using namespace vespa;

const char *const protoOpMessageField = "  {0} {1} = {2};\n";

const std::set<StringRef> expressionOps = {
    "AbsOp",         "BaseClassAddrOp",  "BinOp",          "BitClrsbOp",
    "BitClzOp",      "BitCtzOp",         "BitFfsOp",       "BitParityOp",
    "BitPopcountOp", "ByteswapOp",       "CastOp",         "CeilOp",
    "CmpOp",         "CmpThreeWayOp",    "ComplexBinOp",   "ComplexCreateOp",
    "ComplexImagOp", "ComplexImagPtrOp", "ComplexRealOp",  "ComplexRealPtrOp",
    "ConstantOp",    "CopysignOp",       "CosOp",          "DerivedClassAddrOp",
    "DynamicCastOp", "EhTypeIdOp",       "Exp2Op",         "ExpOp",
    "FAbsOp",        "FMaxOp",           "FMinOp",         "FModOp",
    "FloorOp",       "GetGlobalOp",      "GetMemberOp",    "IsConstantOp",
    "IsFPClassOp",   "IterBeginOp",      "IterEndOp",      "LLrintOp",
    "LLroundOp",     "Log10Op",          "Log2Op",         "LogOp",
    "LrintOp",       "LroundOp",         "NearbyintOp",    "ObjSizeOp",
    "PowOp",         "PtrDiffOp",        "PtrMaskOp",      "PtrStrideOp",
    "RintOp",        "RotateOp",         "RoundOp",        "SelectOp",
    "ShiftOp",       "SignBitOp",        "SinOp",          "SqrtOp",
    "TruncOp",       "UnaryOp",          "VTTAddrPointOp", "VTableAddrPointOp",
};

const std::set<StringRef> instructionOps = {
    "AllocExceptionOp",
    "AllocaOp",
    "AssumeAlignedOp",
    "AssumeOp",
    "AssumeSepStorageOp",
    "AtomicCmpXchg",
    "AtomicFetch",
    "AtomicXchg",
    "BinOpOverflowOp",
    "BrCondOp",
    "BrOp",
    "CIR_InlineAsmOp",
    "CallOp",
    "CatchParamOp",
    "ClearCacheOp",
    "CopyOp",
    "EhInflightOp",
    "ExpectOp",
    "FreeExceptionOp",
    "GetBitfieldOp",
    "GetMethodOp",
    "GetRuntimeMemberOp",
    "LLVMIntrinsicCallOp",
    "LoadOp",
    "MemChrOp",
    "MemCpyInlineOp",
    "MemCpyOp",
    "MemMoveOp",
    "MemSetInlineOp",
    "MemSetOp",
    "PrefetchOp",
    "ResumeOp",
    "ReturnOp",
    "SetBitfieldOp",
    "StackRestoreOp",
    "StackSaveOp",
    "StdFindOp",
    "StoreOp",
    "SwitchFlatOp",
    "ThrowOp",
    "TrapOp",
    "TryCallOp",
    "UnreachableOp",
    "VAArgOp",
    "VACopyOp",
    "VAEndOp",
    "VAStartOp",
    "VecCmpOp",
    "VecCreateOp",
    "VecExtractOp",
    "VecInsertOp",
    "VecShuffleDynamicOp",
    "VecShuffleOp",
    "VecSplatOp",
    "VecTernaryOp",
};

static std::string fixPrefix(const std::string &name) {
  StringRef ref(name);
  std::string result = "";
  if (ref.starts_with("::mlir")) {
    result += "MLIR";
    result += ref.drop_front(8);
  } else if (ref.starts_with("::cir")) {
    result += "CIR";
    result += ref.drop_front(7);
  } else {
    result = ref;
  }
  return result;
}

static std::set<StringRef> brokenEnums = {
  "CIRCallingConvAttr",
  "CIRCatchParamKindAttr",
  "CIRGlobalLinkageKindAttr",
  "CIRTLS_ModelAttr",
  "CIRMemOrderAttr",
};

static bool isEnum(Attribute attr) {
  if (attr.isEnumAttr()) {
    return true;
  }
  auto name = fixPrefix(attr.getStorageType().str());
  return brokenEnums.count(name);
}

static std::string getAttrProtoType(Attribute attr) {
  if (isEnum(attr)) {
    assert(attr.getStorageType().ends_with("Attr"));
    auto name = attr.getStorageType().drop_back(4);
    auto nameFixed = fixPrefix(name.str());
    if (nameFixed == "CIRTLS_Model") {
      return "CIRTLSModel";
    }
    return nameFixed;
  }
  auto name = fixPrefix(attr.getStorageType().str());
  if (name == "MLIRTypedAttr") {
    return "MLIRAttribute";
  }
  return name;
}

static std::string normalizeName(StringRef name) {
  if (!name.ends_with("Op")) {
    return name.str() + "Op";
  }
  return name.str();
}

enum class ValueType {
  REG,
  OPT,
  VAR,
  VAROFVAR,
};

static void serializeValueField(StringRef name, ValueType type, llvm::raw_ostream &os) {
  auto snake = llvm::convertToSnakeFromCamelCase(name);
  auto upperCamel = llvm::convertToCamelFromSnakeCase(name, true);
  switch (type) {
  case ValueType::REG: {
    os << formatv("  *serialized.mutable_{0}() = serializeValue(op.get{1}());\n", snake, upperCamel);
    return;
  }
  case ValueType::OPT: {
    os << formatv("  if (op.get{0}()) {{\n", upperCamel);
    os << formatv("    *serialized.mutable_{0}() = serializeValue(op.get{1}());\n", snake, upperCamel);
    os << formatv("  }\n");
    return;
  }
  case ValueType::VAR: {
    os << formatv("  for (auto v : op.get{0}()) {{\n", upperCamel);
    os << formatv("    auto protoV = serialized.add_{0}();\n", snake);
    os << formatv("    *protoV = serializeValue(v);\n");
    os << formatv("  }\n");
    return;
  }
  case ValueType::VAROFVAR: {
    os << formatv("  for (auto v : op.get{0}()) {{\n", upperCamel);
    os << formatv("    auto protoV = serialized.add_{0}();\n", snake);
    os << formatv("    for (auto vv : v) {{\n");
    os << formatv("      auto protoVV = protoV->add_list();\n");
    os << formatv("      *protoVV = serializeValue(vv);\n");
    os << formatv("    }\n");
    os << formatv("  }\n");
    return;
  }
  }
}

static void serializeResultField(StringRef name, ValueType type, llvm::raw_ostream &os) {
  auto snake = llvm::convertToSnakeFromCamelCase(name);
  auto upperCamel = llvm::convertToCamelFromSnakeCase(name, true);
  switch (type) {
  case ValueType::REG: {
    os << formatv("  *serialized.mutable_{0}() = typeCache.getMLIRTypeID(op.get{1}().getType());\n", snake, upperCamel);
    return;
  }
  case ValueType::OPT: {
    os << formatv("  if (op.get{0}()) {{\n", upperCamel);
    os << formatv("    *serialized.mutable_{0}() = typeCache.getMLIRTypeID(op.get{1}().getType());\n", snake, upperCamel);
    os << formatv("  }\n");
    return;
  }
  case ValueType::VAR: {
    os << formatv("  for (auto t : op.get{0}()) {{\n", upperCamel);
    os << formatv("    auto protoT = serialized.add_{0}();\n", snake);
    os << formatv("    *protoT = typeCache.getMLIRTypeID(t.getType());\n");
    os << formatv("  }\n");
    return;
  }
  case ValueType::VAROFVAR: {
    os << formatv("  for (auto t : op.get{0}()) {{\n", upperCamel);
    os << formatv("    auto protoT = serialized.add_{0}();\n", snake);
    os << formatv("    for (auto tt : t) {{\n");
    os << formatv("      auto protoTT = protoT->add_list();\n");
    os << formatv("      *protoTT = typeCache.getMLIRTypeID(tt.getType());\n");
    os << formatv("    }\n");
    os << formatv("  }\n");
    return;
  }
  }
}

static bool emitOpProto(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << protoHeader;
  os << "\n";
  os << "import \"setup.proto\";\n";
  os << "import \"enum.proto\";\n";
  os << "import \"attr.proto\";\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);
  os << "message MLIROp {\n";
  os << "  MLIROpID id = 1;\n";
  os << "  MLIRLocation location = 2;\n";
  os << "  oneof operation {\n";

  int index = 3;
  for (auto *def : defs) {
    Operator op(*def);
    auto name = normalizeName(op.getCppClassName());
    os << formatv("    CIR{0} {1} = {2};\n", name,
                  llvm::convertToSnakeFromCamelCase(name), index++);
  }
  os << "  }\n";
  os << "}\n";
  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    auto name = normalizeName(op.getCppClassName());

    os << formatv("message CIR{0} {{\n", name);
    size_t messageIdx = 1;
    os << formatv("  // {0} operands\n", op.getNumOperands());
    for (int i = 0; i != op.getNumOperands(); ++i) {
      const auto &operand = op.getOperand(i);
      const auto &operandName = llvm::convertToSnakeFromCamelCase(operand.name);
      if (operand.isOptional()) {
        os << formatv(protoOpMessageField, "optional MLIRValue", operandName,
                      messageIdx++);
      } else if (operand.isVariadicOfVariadic()) {
        os << formatv(protoOpMessageField, "repeated MLIRValueList", operandName,
                      messageIdx++);
      } else if (operand.isVariadic()) {
        os << formatv(protoOpMessageField, "repeated MLIRValue", operandName,
                      messageIdx++);
      } else {
        os << formatv(protoOpMessageField, "MLIRValue", operandName,
                      messageIdx++);
      }
    }
    if (op.getNumOperands() > 0) {
      os << "\n";
    }
    os << formatv("  // {0} native attributes\n", op.getNumNativeAttributes());
    for (int i = 0; i != op.getNumNativeAttributes(); ++i) {
      const auto &attr = op.getAttribute(i).attr;

      const auto &attrName =
        llvm::convertToSnakeFromCamelCase(op.getAttribute(i).name);

      auto protoType = getAttrProtoType(attr);

      if (protoType == "MLIRTypedAttr") {
        protoType = "MLIRAttribute";
      }

      if (attr.getStorageType().starts_with("::cir::AST")) {
        os << formatv("  // [{0} {1}] is ignored\n", protoType, attrName);
        continue;
      }

      if (attr.isOptional()) {
        os << formatv(protoOpMessageField, formatv("optional {0}", protoType),
                      attrName, messageIdx++);
      } else {
        os << formatv(protoOpMessageField, protoType, attrName, messageIdx++);
      }
    }
    if (op.getNumNativeAttributes() > 0) {
      os << "\n";
    }
    os << formatv("  // {0} successors\n", op.getNumSuccessors());
    for (unsigned i = 0; i != op.getNumSuccessors(); ++i) {
      const NamedSuccessor &successor = op.getSuccessor(i);
      const auto &successorName =
          llvm::convertToSnakeFromCamelCase(successor.name);
      if (successor.isVariadic()) {
        os << formatv(protoOpMessageField, "repeated MLIRBlockID", successorName,
                      messageIdx++);
      } else {
        os << formatv(protoOpMessageField, "MLIRBlockID", successorName,
                      messageIdx++);
      }
    }
    if (op.getNumSuccessors() > 0) {
      os << "\n";
    }
    os << formatv("  // {0} results\n", op.getNumResults());
    for (int i = 0; i != op.getNumResults(); ++i) {
      const auto &result = op.getResult(i);
      const auto &resultName = llvm::convertToSnakeFromCamelCase(result.name);
      if (result.isOptional()) {
        os << formatv(protoOpMessageField, "optional MLIRTypeID", resultName,
                      messageIdx++);
      } else if (result.isVariadicOfVariadic()) {
        os << formatv(protoOpMessageField, "repeated MLIRTypeIDArray", resultName,
                      messageIdx++);
      } else if (result.isVariadic()) {
        os << formatv(protoOpMessageField, "repeated MLIRTypeID", resultName,
                      messageIdx++);
      } else {
        os << formatv(protoOpMessageField, "MLIRTypeID", resultName,
                      messageIdx++);
      }
    }

    if (op.getNumRegions() > 0) {
      os << "\n";
      os << formatv("  // {0} regions are ignored for now\n", op.getNumRegions());
    }

    os << formatv("}\n");
    os << formatv("\n");
  }

  os << clangOn;
  return false;
}

static bool emitOpProtoSerializerHeader(const RecordKeeper &records,
                                        raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << "\n";

  os << "#pragma once\n";
  os << "\n";

  os << "#include \"AttrSerializer.h\"\n";
  os << "#include \"Util.h\"\n";
  os << "#include \"proto/op.pb.h\"\n";
  os << "#include \"proto/setup.pb.h\"\n";
  os << "\n";
  os << "#include <clang/CIR/Dialect/IR/CIRDialect.h>\n";
  os << "#include <mlir/IR/Block.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  os << "class OpSerializer {\n";
  os << "public:\n";
  os << "  OpSerializer(MLIRModuleID moduleID,\n";
  os << "               TypeCache &typeCache,\n";
  os << "               OpCache &opCache,\n";
  os << "               BlockCache &blockCache)\n";
  os << "  : moduleID(moduleID), typeCache(typeCache),\n";
  os << "    opCache(opCache), blockCache(blockCache),\n";
  os << "    attributeSerializer(moduleID, typeCache) {}\n";
  os << "\n";

  os << "  MLIROp serializeOperation(mlir::Operation &op);\n";
  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    os << formatv("  CIR{0} serialize{0}(cir::{1} op);\n",
                  normalizeName(op.getCppClassName()), op.getCppClassName());
  }
  os << "\n";

  os << "private:\n";
  os << "  MLIRModuleID moduleID;\n";
  os << "  TypeCache &typeCache;\n";
  os << "  OpCache &opCache;\n";
  os << "  BlockCache &blockCache;\n";
  os << "\n";
  os << "  AttributeSerializer attributeSerializer;\n";
  os << "\n";
  os << "  MLIRValue serializeValue(mlir::Value value);\n";
  os << "};\n";
  os << "\n";

  os << clangOn;
  return false;
}

static bool emitOpProtoSerializerSource(const RecordKeeper &records,
                                        raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << "\n";

  os << "#include \"cir-tac/OpSerializer.h\"\n";
  os << "#include \"cir-tac/EnumSerializer.h\"\n";
  os << "\n";
  os << "#include <llvm/ADT/TypeSwitch.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  os << "MLIROp OpSerializer::serializeOperation(mlir::Operation &op) {\n";
  os << "  MLIROp pOp;\n";
  os << "  *pOp.mutable_location() = attributeSerializer.serializeMLIRLocation(op.getLoc());\n";
  os << "\n";
  os << "  llvm::TypeSwitch<mlir::Operation *>(&op)\n";

  for (auto *def : defs) {
    Operator op(def);
    auto name = normalizeName(op.getCppClassName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);

    os << formatv("  .Case<cir::{0}>([this, &pOp](cir::{0} op) {{\n", op.getCppClassName());
    os << formatv("    auto serialized = serialize{0}(op);\n", name);
    os << formatv("    *pOp.mutable_{0}() = serialized;\n", nameSnake);
    os << formatv("  })\n");
  }

  os << "  .Default([](mlir::Operation *op) {\n";
  os << "    op->dump();\n";
  os << "    llvm_unreachable(\"unknown operation during serialization\");\n";
  os << "  });\n";
  os << "\n";
  os << "  return pOp;\n";
  os << "}\n";

  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    auto name = normalizeName(op.getCppClassName());

    os << formatv("CIR{0} OpSerializer::serialize{0}(cir::{1} op) {{\n", name,
                  op.getCppClassName());
    os << formatv("  CIR{0} serialized;\n", name);
    os << "\n";

    for (int i = 0; i != op.getNumOperands(); ++i) {
      const auto &operand = op.getOperand(i);
      if (operand.isOptional()) {
        serializeValueField(operand.name, ValueType::OPT, os);
      } else if (operand.isVariadicOfVariadic()) {
        serializeValueField(operand.name, ValueType::VAROFVAR, os);
      } else if (operand.isVariadic()) {
        serializeValueField(operand.name, ValueType::VAR, os);
      } else {
        serializeValueField(operand.name, ValueType::REG, os);
      }
    }

    if (op.getNumOperands() > 0) {
      os << "\n";
    }

    for (int i = 0; i != op.getNumNativeAttributes(); ++i) {
      const auto &attr = op.getAttribute(i).attr;

      const auto &attrName =
        llvm::convertToSnakeFromCamelCase(op.getAttribute(i).name);

      const auto &upperCamel = llvm::convertToCamelFromSnakeCase(attrName, true);

      auto protoType = getAttrProtoType(attr);

      if (protoType == "MLIRTypedAttr") {
        protoType = "MLIRAttribute";
      }

      if (attr.getStorageType().starts_with("::cir::AST")) {
        continue;
      }

      if (isEnum(attr) && attr.isOptional()) {
        os << formatv("  if (op.get{0}()) {{\n", upperCamel);
        os << formatv("    serialized.set_{0}(serialize{1}(*op.get{2}()));\n", attrName, protoType, upperCamel);
        os << formatv("  }\n");
      } else if (isEnum(attr)) {
        os << formatv("  serialized.set_{0}(serialize{1}(op.get{2}()));\n", attrName, protoType, upperCamel);
      } else if (attr.isOptional()) {
        os << formatv("  if (op.get{0}Attr()) {{\n", upperCamel);
        os << formatv("    *serialized.mutable_{0}() = attributeSerializer.serialize{1}(op.get{2}Attr());\n",
                      attrName, protoType, upperCamel);
        os << formatv("  }\n");
      } else {
        os << formatv("  *serialized.mutable_{0}() = attributeSerializer.serialize{1}(op.get{2}Attr());\n",
                      attrName, protoType, upperCamel);
      }
    }

    if (op.getNumNativeAttributes() > 0) {
      os << "\n";
    }

    for (unsigned i = 0; i != op.getNumSuccessors(); ++i) {
      const NamedSuccessor &successor = op.getSuccessor(i);
      const auto &snakeName =
        llvm::convertToSnakeFromCamelCase(successor.name);
      const auto &upperCamel = llvm::convertToCamelFromSnakeCase(snakeName, true);
      if (successor.isVariadic()) {
        os << formatv("  for (auto s : op.get{0}()) {{\n", upperCamel);
        os << formatv("    auto protoS = serialized.add_{0}();\n", snakeName);
        os << formatv("    *protoS = blockCache.getMLIRBlockID(s);\n");
        os << formatv("  }\n");
      } else {
        os << formatv("  *serialized.mutable_{0}() = blockCache.getMLIRBlockID(op.get{1}());\n",
                      snakeName, upperCamel);
      }
    }

    if (op.getNumSuccessors() > 0) {
      os << "\n";
    }

    for (int i = 0; i != op.getNumResults(); ++i) {
      const auto &result = op.getResult(i);
      if (result.isOptional()) {
        serializeResultField(result.name, ValueType::OPT, os);
      } else if (result.isVariadicOfVariadic()) {
        serializeResultField(result.name, ValueType::VAROFVAR, os);
      } else if (result.isVariadic()) {
        serializeResultField(result.name, ValueType::VAR, os);
      } else {
        serializeResultField(result.name, ValueType::REG, os);
      }
    }

    if (op.getNumResults() > 0) {
      os << "\n";
    }

    os << "  return serialized;\n";
    os << formatv("}\n\n");
  }

  os << "MLIRValue OpSerializer::serializeValue(mlir::Value value) {\n";
  os << "  MLIRValue pValue;\n";
  os << "  auto typeID = typeCache.getMLIRTypeID(value.getType());\n";
  os << "  *pValue.mutable_type() = typeID;\n";
  os << "\n";
  os << "  llvm::TypeSwitch<mlir::Value>(value)\n";
  os << "      .Case<mlir::OpResult>([this, &pValue](mlir::OpResult value) {\n";
  os << "        MLIROpResult opResult;\n";
  os << "        *opResult.mutable_owner() = opCache.getMLIROpID(value.getOwner());\n";
  os << "        opResult.set_result_number(value.getResultNumber());\n";
  os << "        *pValue.mutable_op_result() = opResult;\n";
  os << "      })\n";
  os << "      .Case<mlir::BlockArgument>([this, &pValue](mlir::BlockArgument value) {\n";
  os << "        MLIRBlockArgument blockArgument;\n";
  os << "        *blockArgument.mutable_owner() =\n";
  os << "            blockCache.getMLIRBlockID(value.getOwner());\n";
  os << "        blockArgument.set_arg_number(value.getArgNumber());\n";
  os << "        *pValue.mutable_block_argument() = blockArgument;\n";
  os << "      })\n";
  os << "      .Default([](mlir::Value value) {\n";
  os << "        value.dump();\n";
  os << "        llvm_unreachable(\"Unknown value during serialization\");\n";
  os << "      });\n";
  os << "  return pValue;\n";
  os << "}\n";
  os << "\n";

  os << clangOn;

  return false;
}

static std::string normalizeFieldName(StringRef name) {
  if (name == "val") {
    return "value";
  }
  if (name == "object") {
    return "obj";
  }
  if (name == "method") {
    return "meth";
  }
  return name.str();
}

static std::string normalizeGetter(StringRef name) {
  if (name == "val") {
    return "getVal()";
  }
  if (name == "object") {
    return "getObject()";
  }
  return name.str();
}

static bool emitOpKotlinExprs(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.api.cir.cfg\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  for (auto *def : defs) {
    Operator op(*def);
    if (expressionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      os << formatv("data class CIR{0}Expr(\n", name);
      for (int i = 0; i != op.getNumOperands(); ++i) {
        const auto &operand = op.getOperand(i);
        auto name = normalizeFieldName(operand.name);
        if (operand.isOptional()) {
          os << formatv("    val {0}: MLIRValue?,\n", name);
        } else if (operand.isVariadicOfVariadic()) {
          os << formatv("    val {0}: List<List<MLIRValue>>,\n", name);
        } else if (operand.isVariadic()) {
          os << formatv("    val {0}: List<MLIRValue>,\n", name);
        } else {
          os << formatv("    val {0}: MLIRValue,\n", name);
        }
      }

      for (int i = 0; i != op.getNumNativeAttributes(); ++i) {
        const auto &attr = op.getAttribute(i).attr;
        const auto &attrName = normalizeFieldName(llvm::convertToCamelFromSnakeCase(op.getAttribute(i).name));
        const auto &protoType = getAttrProtoType(attr);

        if (attr.getStorageType().starts_with("::cir::AST")) {
          continue;
        }

        if (attr.isOptional()) {
          os << formatv("    val {0}: {1}?,\n", attrName, protoType);
        } else {
          os << formatv("    val {0}: {1},\n", attrName, protoType);
        }
      }

      for (unsigned i = 0; i != op.getNumSuccessors(); ++i) {
        const NamedSuccessor &successor = op.getSuccessor(i);
        const auto &successorName = normalizeFieldName(
            llvm::convertToCamelFromSnakeCase(successor.name));
        if (successor.isVariadic()) {
          os << formatv("    val {0}: List<MLIRBlockID>,\n", successorName);
        } else {
          os << formatv("    val {0}: MLIRBlockID,\n", successorName);
        }
      }

      assert(op.getNumResults() == 1);
      auto result = op.getResult(0);
      assert(!result.isOptional() && !result.isVariadic() && !result.isVariadicOfVariadic());
      auto resultName = normalizeFieldName(llvm::convertToCamelFromSnakeCase(result.name));
      os << formatv("    val {0}: MLIRTypeID,\n", resultName);

      os << "): CIRExpr {\n";

      os << formatv("    override val type: MLIRTypeID\n");
      os << formatv("        get() = {0}\n", resultName);

      os << "}\n";

      os << "\n";
    }
  }

  return false;
}

static bool emitOpKotlinInst(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.api.cir.cfg\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  os << "interface CIRInst : CommonInst {\n";
  os << "    override val location: CIRInstLocation,\n";
  os << "    val id: MLIROpID\n";
  os << "}\n";
  os << "\n";

  os <<  "data class CIRAssignInst(\n";
  os <<  "    override val location: CIRInstLocation,\n";
  os <<  "    override val id: MLIROpID,\n";
  os <<  "    override val lhv: MLIRValue,\n";
  os <<  "    override val rhv: CIRExpr,\n";
  os <<  "): CIRInst, CommonAssignInst\n";
  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    if (instructionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      os << formatv("data class CIR{0}Inst(\n", name);
      os <<  "    override val location: CIRInstLocation,\n";
      os <<  "    override val id: MLIROpID,\n";
      os << "\n";
      for (int i = 0; i != op.getNumOperands(); ++i) {
        const auto &operand = op.getOperand(i);
        auto name = normalizeFieldName(operand.name);
        if (operand.isOptional()) {
          os << formatv("    val {0}: MLIRValue?,\n", name);
        } else if (operand.isVariadicOfVariadic()) {
          os << formatv("    val {0}: List<List<MLIRValue>>,\n", name);
        } else if (operand.isVariadic()) {
          os << formatv("    val {0}: List<MLIRValue>,\n", name);
        } else {
          os << formatv("    val {0}: MLIRValue,\n", name);
        }
      }

      if (op.getNumOperands() > 0) {
        os << "\n";
      }

      for (int i = 0; i != op.getNumNativeAttributes(); ++i) {
        const auto &attr = op.getAttribute(i).attr;
        const auto &attrName = normalizeFieldName(llvm::convertToCamelFromSnakeCase(op.getAttribute(i).name));
        const auto &protoType = getAttrProtoType(attr);

        if (attr.getStorageType().starts_with("::cir::AST")) {
          continue;
        }

        if (attr.isOptional()) {
          os << formatv("    val {0}: {1}?,\n", attrName, protoType);
        } else {
          os << formatv("    val {0}: {1},\n", attrName, protoType);
        }
      }

      if (op.getNumNativeAttributes() > 0) {
        os << "\n";
      }

      for (unsigned i = 0; i != op.getNumSuccessors(); ++i) {
        const NamedSuccessor &successor = op.getSuccessor(i);
        const auto &successorName =
          normalizeFieldName(llvm::convertToCamelFromSnakeCase(successor.name));
        if (successor.isVariadic()) {
          os << formatv("    val {0}: List<MLIRBlockID>,\n", successorName);
        } else {
          os << formatv("    val {0}: MLIRBlockID,\n", successorName);
        }
      }

      if (op.getNumSuccessors() > 0) {
        os << "\n";
      }

      for (int i = 0; i != op.getNumResults(); ++i) {
        const auto &result = op.getResult(i);
        auto resultName = normalizeFieldName(llvm::convertToCamelFromSnakeCase(result.name));
        if (result.isOptional()) {
          os << formatv("    val {0}: MLIRTypeID?,\n", resultName);
        } else if (result.isVariadicOfVariadic()) {
          os << formatv("    val {0}: List<List<MLIRTypeID>>,\n", resultName);
        } else if (result.isVariadic()) {
          os << formatv("    val {0}: List<MLIRTypeID>,\n", resultName);
        } else {
          os << formatv("    val {0}: MLIRTypeID,\n", resultName);
        }
      }

      os << "): CIRInst\n";
      os << "\n";
    }
  }

  return false;
}

static bool emitOpKotlinExprsBuilder(const RecordKeeper &records,
                                     raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit. */\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  os << "fun buildExpr(expr: Op.MLIROp) = when (expr.operationCase!!) {\n";
  for (auto *def : defs) {
    Operator op(*def);
    if (expressionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      auto snake = llvm::convertToSnakeFromCamelCase(name);
      auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
      std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
      os << formatv(
          "    Op.MLIROp.OperationCase.{0} -> buildCIR{1}Expr(expr.{2})\n",
                    snake, name, lowerCamel);
    }
  }
  os << "    else -> throw Exception()\n";
  os << "}\n";

  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    if (expressionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      os << formatv("fun buildCIR{0}Expr(expr: Op.CIR{0}) =\n", name);
      os << formatv("    CIR{0}Expr(\n", name);
      for (auto operand : op.getOperands()) {
        auto lowerCamel = normalizeGetter(llvm::convertToCamelFromSnakeCase(operand.name, false));
        auto upperCamel = lowerCamel;
        upperCamel[0] = ::toupper(upperCamel[0]);
        if (operand.isOptional()) {
          os << formatv("        if (expr.has{1}()) buildMLIRValue(expr.{0}) else null,\n", lowerCamel, upperCamel);
        } else if (operand.isVariadicOfVariadic()) {
          os << formatv("        buildMLIRValueArrayArray(expr.{0}List),\n", lowerCamel);
        } else if (operand.isVariadic()) {
          os << formatv("        buildMLIRValueArray(expr.{0}List),\n", lowerCamel);
        } else {
          os << formatv("        buildMLIRValue(expr.{0}),\n", lowerCamel);
        }
      }

      for (int i = 0; i < op.getNumNativeAttributes(); ++i) {
        const auto &attr = op.getAttribute(i).attr;
        const auto &attrName = normalizeGetter(llvm::convertToCamelFromSnakeCase(op.getAttribute(i).name));
        auto upperCamel = llvm::convertToCamelFromSnakeCase(attrName, true);
        const auto &protoType = getAttrProtoType(attr);

        if (attr.getStorageType().starts_with("::cir::AST")) {
          continue;
        }

        if (attr.isOptional()) {
          os << formatv("        if (expr.has{2}()) build{1}(expr.{0}) else null,\n", attrName, protoType, upperCamel);
        } else {
          os << formatv("        build{1}(expr.{0}),\n", attrName, protoType);
        }
      }

      for (auto successor : op.getSuccessors()) {
        const auto &successorName =
          normalizeGetter(llvm::convertToCamelFromSnakeCase(successor.name));
        if (successor.isVariadic()) {
          os << formatv("        buildMLIRBlockIDArray(expr.{0}List),\n", successorName);
        } else {
          os << formatv("        buildMLIRBlockID(expr.{0}),\n", successorName);
        }
      }

      for (auto result : op.getResults()) {
        auto lowerCamel = normalizeGetter(llvm::convertToCamelFromSnakeCase(result.name, false));
        auto upperCamel = lowerCamel;
        upperCamel[0] = ::toupper(upperCamel[0]);
        if (result.isOptional()) {
          os << formatv("        if (expr.has{0}()) buildMLIRTypeID(expr.{1}) else null,\n", upperCamel, lowerCamel);
        } else if (result.isVariadicOfVariadic()) {
          os << formatv("        buildMLIRTypeIDArrayArray(expr.{0}List),\n", lowerCamel);
        } else if (result.isVariadic()) {
          os << formatv("        buildMLIRTypeIDArray(expr.{0}List),\n", lowerCamel);
        } else {
          os << formatv("        buildMLIRTypeID(expr.{0}),\n", lowerCamel);
        }
      }

      os << ")\n";
      os << "\n";
    }
  }

  return false;
}

static bool emitOpKotlinInstBuilder(const RecordKeeper &records,
                                     raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit. */\n";
  os << "\n";

  auto defs = getRequestedOpDefinitions(records);

  os << "private val CIRExpressions = listOf(\n";
  for (auto exprOp : expressionOps) {
    auto snake = llvm::convertToSnakeFromCamelCase(exprOp);
    std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
    os << formatv("    Op.MLIROp.OperationCase.{0},\n", snake);
  }
  os << ")\n";
  os << "\n";

  os << "class CIRInstBuilder(private val function: CIRFunction) {\n";
  os << "\n";
  os << "    fun buildInst(inst: Op.MLIROp) : CIRInst {\n";
  os << "        val id = buildMLIROpID(inst.id)\n";
  os << "        val location = CIRInstLocation(function, buildMLIRLocation(inst.location))\n";
  os << "\n";

  os << "        if (CIRExpressions.contains(inst.operationCase!!)) {\n";
  os << "            val expr = buildExpr(inst)\n";
  os << "            return CIRAssignInst(\n";
  os << "                location,\n";
  os << "                id,\n";
  os << "                MLIROpValue(expr.type, buildMLIROpID(inst.id), 0),\n";
  os << "                expr\n";
  os << "            )\n";
  os << "        }\n";
  os << "\n";

  os << "        return when (inst.operationCase!!) {\n";

  for (auto *def : defs) {
    Operator op(*def);
    if (instructionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      auto snake = llvm::convertToSnakeFromCamelCase(name);
      auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
      std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
      os << formatv(
                    "            Op.MLIROp.OperationCase.{0} -> buildCIR{1}Inst(inst.{2}, id, location)\n",
                    snake, name, lowerCamel);
    }
  }
  os << "            else -> throw Exception()\n";
  os << "        }\n";
  os << "    }\n";
  os << "}\n";
  os << "\n";

  for (auto *def : defs) {
    Operator op(*def);
    if (instructionOps.count(op.getCppClassName())) {
      auto name = normalizeName(op.getCppClassName());
      os << formatv("fun buildCIR{0}Inst(inst: Op.CIR{0}, id: MLIROpID, location: CIRInstLocation) =\n", name);
      os << formatv("    CIR{0}Inst(\n", name);
      os << formatv("        location,\n");
      os << formatv("        id,\n");
      for (auto operand : op.getOperands()) {
        auto lowerCamel = normalizeGetter(llvm::convertToCamelFromSnakeCase(operand.name, false));
        auto upperCamel = lowerCamel;
        upperCamel[0] = ::toupper(upperCamel[0]);
        if (operand.isOptional()) {
          os << formatv("        if (inst.has{1}()) buildMLIRValue(inst.{0}) else null,\n", lowerCamel, upperCamel);
        } else if (operand.isVariadicOfVariadic()) {
          os << formatv("        buildMLIRValueArrayArray(inst.{0}List),\n", lowerCamel);
        } else if (operand.isVariadic()) {
          os << formatv("        buildMLIRValueArray(inst.{0}List),\n", lowerCamel);
        } else {
          os << formatv("        buildMLIRValue(inst.{0}),\n", lowerCamel);
        }
      }

      for (int i = 0; i < op.getNumNativeAttributes(); ++i) {
        const auto &attr = op.getAttribute(i).attr;
        const auto &attrName = normalizeGetter(llvm::convertToCamelFromSnakeCase(op.getAttribute(i).name));
        auto upperCamel = llvm::convertToCamelFromSnakeCase(attrName, true);
        const auto &protoType = getAttrProtoType(attr);

        if (attr.getStorageType().starts_with("::cir::AST")) {
          continue;
        }

        if (attr.isOptional()) {
          os << formatv("        if (inst.has{2}()) build{1}(inst.{0}) else null,\n", attrName, protoType, upperCamel);
        } else {
          os << formatv("        build{1}(inst.{0}),\n", attrName, protoType);
        }
      }

      for (auto successor : op.getSuccessors()) {
        const auto &successorName =
          normalizeGetter(llvm::convertToCamelFromSnakeCase(successor.name));
        if (successor.isVariadic()) {
          os << formatv("        buildMLIRBlockIDArray(inst.{0}List),\n", successorName);
        } else {
          os << formatv("        buildMLIRBlockID(inst.{0}),\n", successorName);
        }
      }

      for (auto result : op.getResults()) {
        auto lowerCamel = normalizeGetter(llvm::convertToCamelFromSnakeCase(result.name, false));
        auto upperCamel = lowerCamel;
        upperCamel[0] = ::toupper(upperCamel[0]);
        if (result.isOptional()) {
          os << formatv("        if (inst.has{0}()) buildMLIRTypeID(inst.{1}) else null,\n", upperCamel, lowerCamel);
        } else if (result.isVariadicOfVariadic()) {
          os << formatv("        buildMLIRTypeIDArrayArray(inst.{0}List),\n", lowerCamel);
        } else if (result.isVariadic()) {
          os << formatv("        buildMLIRTypeIDArray(inst.{0}List),\n", lowerCamel);
        } else {
          os << formatv("        buildMLIRTypeID(inst.{0}),\n", lowerCamel);
        }
      }

      os << ")\n";
      os << "\n";
    }
  }

  return false;
}

static mlir::GenRegistration genOpProto("gen-op-proto",
                                        "Generate proto file for ops",
                                        &emitOpProto);

static mlir::GenRegistration
    genOpProtoSerializerHeader("gen-op-proto-serializer-header",
                               "Generate proto serializer .h for ops",
                               &emitOpProtoSerializerHeader);

static mlir::GenRegistration
    genOpProtoSerializerSource("gen-op-proto-serializer-source",
                               "Generate proto serializer .cpp for ops",
                               &emitOpProtoSerializerSource);

static mlir::GenRegistration genOpKotlinExprs("gen-op-kotlin-expr",
                                        "Generate kotlin expr for ops",
                                        &emitOpKotlinExprs);

static mlir::GenRegistration genOpKotlinInst("gen-op-kotlin-inst",
                                        "Generate kotlin inst for ops",
                                        &emitOpKotlinInst);

static mlir::GenRegistration genOpKotlinExprsBuilder("gen-op-kotlin-expr-builder",
                                        "Generate kotlin expr builder for ops",
                                                     &emitOpKotlinExprsBuilder);

static mlir::GenRegistration genOpKotliInstBuilder("gen-op-kotlin-inst-builder",
                                        "Generate kotlin inst builder for ops",
                                        &emitOpKotlinInstBuilder);
