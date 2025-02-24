#include "VespaCommon.h"
#include "VespaGen.h"

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

#include <set>

using llvm::formatv;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::AttrDef;
using mlir::tblgen::EnumAttr;

using namespace vespa;

std::set<StringRef> mlirAttributeWhitelist = {
    "ArrayAttr", "StringAttr",     "IntegerAttr", "FloatAttr",
    "TypeAttr",  "DictionaryAttr", "UnitAttr",
};

std::string normalizeName(StringRef name) {

  if (name == "ExtraFuncAttr") {
    return "ExtraFuncAttributesAttr";
  }

  if (name.starts_with("Builtin_")) {
    name = name.drop_front(8);
  } else if (name.starts_with("CIR_")) {
    name = name.drop_front(4);
  }

  std::string result = name.str();
  if (!name.ends_with("Loc") && !name.ends_with("Attr")) {
    result += "Attr";
  }
  return result;
}

static std::string normalizeEnumName(llvm::StringRef name) {
  if (name == "TLS_Model") {
    return normalizeName("TLSModel");
  }
  return normalizeName(name.str());
}

static bool emitAttrProto(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << jacoDBLicense;
  os << protoHeader;
  os << "\n";

  os << "import \"setup.proto\";\n";
  os << "import \"enum.proto\";\n";
  os << "\n";

  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");

  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");

  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");

  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  // MLIRAttribute message generation
  os << "message MLIRAttribute {\n";
  os << "  oneof attribute {\n";
  size_t count = 1;

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    if (mlirAttributeWhitelist.count(name)) {
      os << formatv("    MLIR{0} {1} = {2};\n", name, nameSnake, count++);
    }
  }

  // These attributs are needed but absent from the .td file
  os << formatv("    MLIRNamedAttr named_attr = {0};\n", count++);
  os << formatv("    MLIRFlatSymbolRefAttr flat_symbol_ref_attr = {0};\n",
                count++);
  os << formatv("    MLIRDenseI32ArrayAttr dense_i32_array_attr = {0};\n",
                count++);

  // Location attributes are a subclass of attributes
  os << "\n";
  os << formatv("    MLIRLocation location = {0};\n", count++);
  os << "\n";

  for (auto *def : cirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);

    if (StringRef(name).starts_with("AST")) {
      // AST attributes are not supported
      continue;
    }

    os << formatv("    CIR{0} {1} = {2};\n", name, nameSnake, count++);
  }

  os << "\n";

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("    CIR{0} {1} = {2};\n", name, nameSnake, count++);
  }

  os << "  }\n";
  os << "}\n";
  os << "\n";

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (mlirAttributeWhitelist.count(name)) {
      os << formatv("message MLIR{0} {{\n", name);
      size_t index = 1;
      for (auto param : attr.getParameters()) {
        auto paramNameSnake =
            llvm::convertToSnakeFromCamelCase(param.getName());
        if (param.isOptional()) {
          os << formatv("  optional {0} {1} = {2};\n",
                        getProtoType(param), paramNameSnake,
                        index++);
        } else {
          os << formatv("  {0} {1} = {2};\n",
                        getProtoType(param), paramNameSnake,
                        index++);
        }
      }
      os << "}\n";
      os << "\n";
    }
  }

  // These attributes are needed but absent from the .td file

  os << "message MLIRNamedAttr {\n";
  os << "  MLIRStringAttr name = 1;\n";
  os << "  MLIRAttribute value = 2;\n";
  os << "}\n";
  os << "\n";

  os << "message MLIRFlatSymbolRefAttr {\n";
  os << "  MLIRStringAttr root_reference = 1;\n";
  os << "}\n";
  os << "\n";

  os << "message MLIRDenseI32ArrayAttr {\n";
  os << "  int64 size = 1;\n";
  os << "  repeated int32 raw_data = 2;\n";
  os << "}\n";
  os << "\n";

  // Location variant
  os << "message MLIRLocation {\n";
  os << "  oneof location {\n";
  count = 1;
  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("    MLIR{0} {1} = {2};\n", name, nameSnake, count++);
  }
  os << "  }\n";
  os << "}\n";
  os << "\n";

  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    os << formatv("message MLIR{0} {{\n", name);
    size_t index = 1;
    for (auto param : attr.getParameters()) {
      // For OpaqueLoc, can make use of fallbackLocation only
      if (param.getName() == "underlyingLocation" ||
          param.getName() == "underlyingTypeID") {
        continue;
      }

      auto paramNameSnake = llvm::convertToSnakeFromCamelCase(param.getName());
      if (param.isOptional() ||
          (name == "FusedLoc" && param.getName() == "metadata")) {
        os << formatv("  optional {0} {1} = {2};\n", getProtoType(param),
                      paramNameSnake, index++);
      } else {
        os << formatv("  {0} {1} = {2};\n", getProtoType(param), paramNameSnake,
                      index++);
      }
    }
    os << "}\n";
    os << "\n";
  }

  for (auto *def : cirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());

    if (attr.getName().starts_with("AST")) {
      continue;
    }

    os << formatv("message CIR{0} {{\n", name);
    size_t index = 1;
    for (auto param : attr.getParameters()) {
      auto paramNameSnake = llvm::convertToSnakeFromCamelCase(param.getName());
      if (param.isOptional()) {
        os << formatv("  optional {0} {1} = {2};\n", getProtoType(param),
                      paramNameSnake, index++);
      } else {
        os << formatv("  {0} {1} = {2};\n", getProtoType(param), paramNameSnake,
                      index++);
      }
    }
    os << "}\n";
    os << "\n";
  }

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    os << formatv("message CIR{0} {{\n", name);
    // Drop "Attr"
    os << formatv("  CIR{0} value = 1;\n", StringRef(name).drop_back(4));
    os << "}\n";
    os << "\n";
  }

  os << clangOn;
  return false;
}

static bool emitAttrProtoSerializerHeader(const RecordKeeper &records,
                                          raw_ostream &os) {
  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");

  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");

  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");

  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  os << autogenMessage;
  os << clangOff;
  os << jacoDBLicense;
  os << "\n";
  os << "#pragma once\n";
  os << "\n";

  os << "#include \"Util.h\"\n";
  os << "#include \"proto/attr.pb.h\"\n";
  os << "\n";
  os << "#include <clang/CIR/Dialect/IR/CIRAttrs.h>\n";
  os << "#include <mlir/IR/BuiltinAttributes.h>\n";
  os << "#include <mlir/IR/Location.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  os << "class AttributeSerializer {\n";
  os << "public:\n";
  os << "  AttributeSerializer(MLIRModuleID moduleID, TypeCache &typeCache)\n";
  os << "      : moduleID(moduleID), typeCache(typeCache) {}\n";
  os << "\n";

  os << "  MLIRAttribute serializeMLIRAttribute(mlir::Attribute);\n";
  os << "\n";

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());

    if (mlirAttributeWhitelist.count(name)) {
      os << formatv("  MLIR{0} serializeMLIR{0}(mlir::{0});\n", name);
    }
  }

  os << "  MLIRNamedAttr serializeMLIRNamedAttr(mlir::NamedAttribute);\n";
  os << "  MLIRFlatSymbolRefAttr "
        "serializeMLIRFlatSymbolRefAttr(mlir::FlatSymbolRefAttr);\n";
  os << "  MLIRDenseI32ArrayAttr serializeMLIRDenseI32ArrayAttr(mlir::DenseI32ArrayAttr);\n";
  os << "\n";

  os << "  MLIRLocation serializeMLIRLocation(mlir::Location);\n";
  os << "\n";

  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    os << formatv("  MLIR{0} serializeMLIR{0}(mlir::{0});\n", name);
  }
  os << "\n";

  for (auto *def : cirDefs) {
    AttrDef attr(def);

    if (attr.getName().starts_with("AST")) {
      continue;
    }

    auto name = normalizeName(attr.getName());
    os << formatv("  CIR{0} serializeCIR{0}(cir::{0});\n", name);
  }
  os << "\n";

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    auto cppName = name;
    if (name == "TLSModelAttr") {
      cppName = "TLS_ModelAttr";
    }
    os << formatv("  CIR{0} serializeCIR{0}(cir::{1});\n", name, cppName);
  }
  os << "\n";

  os << "private:\n";
  os << "  MLIRModuleID moduleID;\n";
  os << "  TypeCache &typeCache;\n";
  os << "};\n";

  os << "\n";

  os << clangOn;
  return false;
}

static bool emitAttrProtoSerializerSource(const RecordKeeper &records,
                                          raw_ostream &os) {
  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");
  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");
  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");
  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");
  os << autogenMessage;
  os << clangOff;
  os << "\n";
  os << "#include \"cir-tac/AttrSerializer.h\"\n";
  os << "#include \"cir-tac/EnumSerializer.h\"\n";
  os << "#include \"proto/attr.pb.h\"\n";
  os << "\n";
  os << "#include <llvm/ADT/TypeSwitch.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";
  os << "MLIRAttribute "
        "AttributeSerializer::serializeMLIRAttribute(mlir::Attribute attr) {\n";
  os << "  MLIRAttribute pAttr;\n";
  os << "\n";
  os << "  llvm::TypeSwitch<mlir::Attribute>(attr)\n";
  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    if (mlirAttributeWhitelist.count(name)) {
      os << formatv("  .Case<mlir::{0}>([this, &pAttr](mlir::{0} attr) {{\n",
                    name);
      os << formatv("    auto serialized = serializeMLIR{0}(attr);\n", name);
      os << formatv("    *pAttr.mutable_{0}() = serialized;\n", nameSnake);
      os << formatv("  })\n");
    }
  }
  os << "  .Case<mlir::FlatSymbolRefAttr>([this, "
        "&pAttr](mlir::FlatSymbolRefAttr attr) {\n";
  os << "    auto serialized = serializeMLIRFlatSymbolRefAttr(attr);\n";
  os << "    *pAttr.mutable_flat_symbol_ref_attr() = serialized;\n";
  os << "  })\n";
  os << "  .Case<mlir::LocationAttr>([this, &pAttr](mlir::LocationAttr attr) {\n";
  os << "    auto serialized = serializeMLIRLocation(attr);\n";
  os << "    *pAttr.mutable_location() = serialized;\n";
  os << "  })\n";
  for (auto *def : cirDefs) {
    AttrDef attr(def);
    if (attr.getName().starts_with("AST")) {
      continue;
    }
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("  .Case<cir::{0}>([this, &pAttr](cir::{0} attr) {{\n", name);
    os << formatv("    auto serialized = serializeCIR{0}(attr);\n", name);
    os << formatv("    *pAttr.mutable_{0}() = serialized;\n", nameSnake);
    os << formatv("  })\n");
  }
  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    auto cppName = name;
    if (name == "TLSModelAttr") {
      cppName = "TLS_ModelAttr";
    }
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("  .Case<cir::{0}>([this, &pAttr](cir::{0} attr) {{\n", cppName);
    os << formatv("    auto serialized = serializeCIR{0}(attr);\n", name);
    os << formatv("    *pAttr.mutable_{0}() = serialized;\n", nameSnake);
    os << formatv("  })\n");
  }
  os << "  .Default([](mlir::Attribute attr) {\n";
  os << "    attr.dump();\n";
  os << "    llvm_unreachable(\"unknown attribute during serialization\");\n";
  os << "  });\n";
  os << "\n";
  os << "  return pAttr;\n";
  os << "}\n";
  os << "\n";
  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    if (mlirAttributeWhitelist.count(name)) {
      os << formatv(
          "MLIR{0} AttributeSerializer::serializeMLIR{0}(mlir::{0} attr) {{\n",
          name);
      os << formatv("  MLIR{0} serialized;\n", name);
      for (auto param : attr.getParameters()) {
        serializeParameter(param, "attr", os);
      }
      os << "  return serialized;\n";
      os << "}\n";
      os << "\n";
    }
  }
  os << "MLIRNamedAttr AttributeSerializer::serializeMLIRNamedAttr(mlir::NamedAttribute attr) {\n";
  os << "  MLIRNamedAttr serialized;\n";
  os << "  *serialized.mutable_name() = serializeMLIRStringAttr(attr.getName());\n";
  os << "  *serialized.mutable_value() = serializeMLIRAttribute(attr.getValue());\n";
  os << "  return serialized;\n";
  os << "}\n";
  os << "\n";
  os << "MLIRFlatSymbolRefAttr AttributeSerializer::serializeMLIRFlatSymbolRefAttr(mlir::FlatSymbolRefAttr attr) {\n";
  os << "  MLIRFlatSymbolRefAttr serialized;\n";
  os << "  *serialized.mutable_root_reference() = serializeMLIRStringAttr(attr.getRootReference());\n";
  os << "  return serialized;\n";
  os << "}\n";
  os << "\n";
  os << "MLIRDenseI32ArrayAttr AttributeSerializer::serializeMLIRDenseI32ArrayAttr(mlir::DenseI32ArrayAttr attr) {\n";
  os << "  MLIRDenseI32ArrayAttr serialized;\n";
  os << "  serialized.set_size(attr.getSize());\n";
  os << "  for (auto i : attr.getRawData()) {\n";
  os << "    serialized.mutable_raw_data()->Add(i);\n";
  os << "  }\n";
  os << "  return serialized;\n";
  os << "}\n";
  os << "\n";
  os << "MLIRLocation "
        "AttributeSerializer::serializeMLIRLocation(mlir::Location attr) {\n";
  os << "  MLIRLocation pAttr;\n";
  os << "\n";
  os << "  llvm::TypeSwitch<mlir::Location>(attr)\n";
  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("  .Case<mlir::{0}>([this, &pAttr](mlir::{0} attr) {{\n",
                  name);
    os << formatv("    auto serialized = serializeMLIR{0}(attr);\n", name);
    os << formatv("    pAttr.mutable_{0}()->CopyFrom(serialized);\n",
                  nameSnake);
    os << formatv("  })\n");
  }
  os << "  .Default([](mlir::Attribute attr) {\n";
  os << "    attr.dump();\n";
  os << "    llvm_unreachable(\"unknown attribute during serialization\");\n";
  os << "  });\n";
  os << "\n";
  os << "  return pAttr;\n";
  os << "}\n";
  os << "\n";
  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv(
        "MLIR{0} AttributeSerializer::serializeMLIR{0}(mlir::{0} attr) {{\n",
        name);
    os << formatv("  MLIR{0} serialized;\n", name);
    for (auto param : attr.getParameters()) {
      // For OpaqueLoc, can make use of fallbackLocation only
      if (param.getName() == "underlyingLocation" ||
          param.getName() == "underlyingTypeID") {
        continue;
      }
      // This is actually optional although not stated so in the .td file
      if (name == "FusedLoc" && param.getName() == "metadata") {
        os << "  if (attr.getMetadata()) {\n";
        os << "    *serialized.mutable_metadata() = serializeMLIRAttribute(attr.getMetadata());\n";
        os << "  }\n";
        continue;
      }
      serializeParameter(param, "attr", os);
    }
    os << "  return serialized;\n";
    os << "}\n";
    os << "\n";
  }
  for (auto *def : cirDefs) {
    AttrDef attr(def);
    if (attr.getName().starts_with("AST")) {
      continue;
    }
    auto name = normalizeName(attr.getName());
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv(
        "CIR{0} AttributeSerializer::serializeCIR{0}(cir::{0} attr) {{\n",
        name);
    os << formatv("  CIR{0} serialized;\n", name);
    for (auto param : attr.getParameters()) {
      serializeParameter(param, "attr", os);
    }
    os << "  return serialized;\n";
    os << "}\n";
    os << "\n";
  }
  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    auto cppName = name;
    if (name == "TLSModelAttr") {
      cppName = "TLS_ModelAttr";
    }
    auto nameSnake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv(
        "CIR{0} AttributeSerializer::serializeCIR{0}(cir::{1} attr) {{\n",
                  name, cppName);
    os << formatv("  CIR{0} serialized;\n", name);
    if (name == "SignedOverflowBehaviorAttr") {
      os << formatv("  serialized.set_value(serializeCIR{0}(attr.getBehavior()));\n", StringRef(name).drop_back(4));
    } else {
      os << formatv("  serialized.set_value(serializeCIR{0}(attr.getValue()));\n", StringRef(name).drop_back(4));
    }
    os << "  return serialized;\n";
    os << "}\n";
    os << "\n";
  }
  os << clangOn;
  return false;
}

static bool emitAttrKotlin(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.api.cir.cfg\n";
  os << "\n";
  os << "import java.math.BigDecimal\n";
  os << "import java.math.BigInteger\n";
  os << "\n";

  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");

  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");

  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");

  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  os << "interface MLIRAttribute\n";
  os << "\n";

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (mlirAttributeWhitelist.count(name)) {
      if (attr.getNumParameters() == 0) {
        os << formatv("class MLIR{0} : MLIRAttribute\n", name);
      } else {
        os << formatv("data class MLIR{0}(\n", name);
        for (auto param : attr.getParameters()) {
          auto paramName = llvm::convertToCamelFromSnakeCase(param.getName());
          if (param.isOptional()) {
            os << formatv("    val {0}: {1}?,\n", paramName,
                          getKotlinType(param));
          } else {
            os << formatv("    val {0}: {1},\n", paramName, getKotlinType(param));
          }
        }
        os << ") : MLIRAttribute\n";
      }
      os << "\n";
    }
  }

  os << "data class MLIRNamedAttr(\n";
  os << "    val name: MLIRStringAttr,\n";
  os << "    val value: MLIRAttribute,\n";
  os << ") : MLIRAttribute\n";
  os << "\n";
  os << "data class MLIRFlatSymbolRefAttr(\n";
  os << "    val rootReference: MLIRStringAttr,\n";
  os << ") : MLIRAttribute\n";
  os << "\n";
  os << "data class MLIRDenseI32ArrayAttr(\n";
  os << "    val size: Long,\n";
  os << "    val rawData: ArrayList<Int>,\n";
  os << ") : MLIRAttribute\n";
  os << "\n";

  os << "interface MLIRLocation : MLIRAttribute\n";
  os << "\n";

  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (attr.getNumParameters() == 0) {
      os << formatv("class MLIR{0} : MLIRLocation\n", name);
    } else {
      os << formatv("data class MLIR{0}(\n", name);
      for (auto param : attr.getParameters()) {
        // For OpaqueLoc, can make use of fallbackLocation only
        if (param.getName() == "underlyingLocation" ||
            param.getName() == "underlyingTypeID") {
          continue;
        }
        auto paramName = llvm::convertToCamelFromSnakeCase(param.getName());
        if (param.isOptional() || (name == "FusedLoc" && param.getName() == "metadata")) {
          os << formatv("    val {0}: {1}?,\n", paramName, getKotlinType(param));
        } else {
          os << formatv("    val {0}: {1},\n", paramName, getKotlinType(param));
        }
      }
      os << ") : MLIRLocation\n";
    }
    os << "\n";
  }

  for (auto *def : cirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());

    if (attr.getName().starts_with("AST")) {
      continue;
    }

    if (attr.getNumParameters() == 0) {
      os << formatv("class CIR{0} : MLIRAttribute\n", name);
    } else {
      os << formatv("data class CIR{0}(\n", name);
      for (auto param : attr.getParameters()) {
        auto paramName = llvm::convertToCamelFromSnakeCase(param.getName());
        if (param.isOptional()) {
          os << formatv("    val {0}: {1}?,\n", paramName, getKotlinType(param));
        } else {
          os << formatv("    val {0}: {1},\n", paramName, getKotlinType(param));
        }
      }
      os << ") : MLIRAttribute\n";
    }
    os << "\n";
  }

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    os << formatv("data class CIR{0}(\n", name);
    os << formatv("    val value: CIR{0},\n", StringRef(name).drop_back(4));
    os << formatv(") : MLIRAttribute\n");
    os << "\n";
  }

  return false;
}

static bool emitAttrKotlinBuilder(const RecordKeeper &records,
                                  raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.impl.cfg.builder\n";
  os << "\n";
  os << "import org.jacodb.api.cir.cfg.*\n";
  os << "import org.jacodb.impl.grpc.Attr\n";
  os << "\n";
  os << "import java.math.BigDecimal\n";
  os << "import java.math.BigInteger\n";
  os << "\n";

  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");

  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");

  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");

  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  os << "fun buildMLIRAttribute(attr: Attr.MLIRAttribute): MLIRAttribute = when "
        "(attr.attributeCase!!) {\n";

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (mlirAttributeWhitelist.count(name)) {
      auto snake = llvm::convertToSnakeFromCamelCase(name);
      auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
      std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
      os << formatv(
          "    Attr.MLIRAttribute.AttributeCase.{0} -> buildMLIR{1}(attr.{2})\n",
          snake, name, lowerCamel);
    }
  }

  os << "    Attr.MLIRAttribute.AttributeCase.NAMED_ATTR -> "
        "buildMLIRNamedAttr(attr.namedAttr)\n";
  os << "    Attr.MLIRAttribute.AttributeCase.FLAT_SYMBOL_REF_ATTR -> "
        "buildMLIRFlatSymbolRefAttr(attr.flatSymbolRefAttr)\n";
  os << "    Attr.MLIRAttribute.AttributeCase.DENSE_I32_ARRAY_ATTR -> buildMLIRDenseI32ArrayAttr(attr.denseI32ArrayAttr)\n";
  os << "    Attr.MLIRAttribute.AttributeCase.LOCATION -> "
        "buildMLIRLocation(attr.location)\n";

  for (auto *def : cirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());

    if (attr.getName().starts_with("AST")) {
      continue;
    }

    auto snake = llvm::convertToSnakeFromCamelCase(name);
    auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
    std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
    os << formatv(
        "    Attr.MLIRAttribute.AttributeCase.{0} -> buildCIR{1}(attr.{2})\n",
        snake, name, lowerCamel);
  }

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    auto snake = llvm::convertToSnakeFromCamelCase(name);
    auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
    std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
    os << formatv(
        "    Attr.MLIRAttribute.AttributeCase.{0} -> buildCIR{1}(attr.{2})\n",
        snake, name, lowerCamel);
  }

  os << "    Attr.MLIRAttribute.AttributeCase.ATTRIBUTE_NOT_SET -> throw Exception()\n";
  os << "}\n";
  os << "\n";

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (mlirAttributeWhitelist.count(name)) {
      os << formatv("fun buildMLIR{0}(attr: Attr.MLIR{0}) = MLIR{0}(\n", name);
      for (auto param : attr.getParameters()) {
        buildParameter(param, "attr", os, 4);
      }
      os << ")\n";
      os << "\n";
    }
  }

  os << "fun buildMLIRNamedAttr(attr: Attr.MLIRNamedAttr) = MLIRNamedAttr(\n";
  os << "    buildMLIRStringAttr(attr.name),\n";
  os << "    buildMLIRAttribute(attr.value),\n";
  os << ")\n";
  os << "\n";
  os << "fun buildMLIRFlatSymbolRefAttr(attr: Attr.MLIRFlatSymbolRefAttr) = "
        "MLIRFlatSymbolRefAttr(\n";
  os << "    buildMLIRStringAttr(attr.rootReference),\n";
  os << ")\n";
  os << "\n";
  os << "fun buildMLIRDenseI32ArrayAttr(attr: Attr.MLIRDenseI32ArrayAttr) = MLIRDenseI32ArrayAttr(\n";
  os << "    attr.size,\n";
  os << "    buildI32Array(attr.rawDataList),\n";
  os << ")\n";
  os << "\n";

  os << "fun buildMLIRLocation(attr: Attr.MLIRLocation): MLIRLocation = when "
        "(attr.locationCase!!) {\n";

  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    auto snake = llvm::convertToSnakeFromCamelCase(name);
    auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
    std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
    os << formatv(
        "    Attr.MLIRLocation.LocationCase.{0} -> buildMLIR{1}(attr.{2})\n",
        snake, name, lowerCamel);
  }

  os << "    Attr.MLIRLocation.LocationCase.LOCATION_NOT_SET -> throw Exception()\n";
  os << "}\n";
  os << "\n";

  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    os << formatv("fun buildMLIR{0}(attr: Attr.MLIR{0}) = MLIR{0}(\n", name);
    for (auto param : attr.getParameters()) {
      if (param.getName() == "underlyingLocation" ||
          param.getName() == "underlyingTypeID") {
        continue;
      }
      if (name == "FusedLoc" && param.getName() == "metadata") {
        os << "    if (attr.hasMetadata()) buildMLIRAttribute(attr.metadata) else null,\n";
        continue;
      }
      buildParameter(param, "attr", os, 4);
    }
    os << ")\n";
    os << "\n";
  }

  for (auto *def : cirDefs) {
    AttrDef attr(def);
    if (attr.getName().starts_with("AST")) {
      continue;
    }
    auto name = normalizeName(attr.getName());
    os << formatv("fun buildCIR{0}(attr: Attr.CIR{0}) = CIR{0}(\n", name);
    for (auto param : attr.getParameters()) {
      buildParameter(param, "attr", os, 4);
    }
    os << ")\n";
    os << "\n";
  }

  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    auto name = normalizeEnumName(attr.getEnumClassName());
    os << formatv("fun buildCIR{0}(attr: Attr.CIR{0}) = CIR{0}(\n", name);
    os << formatv("    buildCIR{0}(attr.value),\n", StringRef(name).drop_back(4));
    os << formatv(")\n");
    os << "\n";
  }

  return false;
}

std::set<StringRef> attributeNoCtxBuilder = {
    "IntegerAttr", "FloatAttr", "StringAttr", "TypeAttr",
    "AffineMapAttr", "SymbolRefAttr", "CallSiteLoc", "NameLoc",
    "DenseElementsAttr", "DenseElementsAttr", "DenseResourceElementsAttr",
    "IntegerSetAttr", "OpaqueAttr", "SparseElementsAttr"
};

static bool doesNeedCtx(const std::string &name) {
  return !attributeNoCtxBuilder.count(name);
}

inline static void aggregateAttribute(const AttrDef &def,
                                      const CppTypeInfo &cppNamespace,
                                      const std::string &varName,
                                      CppProtoDeserializer &addTo,
                                      bool addAsCase = true) {
  auto name = normalizeName(def.getName());
  std::string cppName = formatv("{0}::{1}", cppNamespace.factualType, name).str();
  auto deserializer = deserializeParameters(name, cppName, def.getParameters(),
    varName, name, doesNeedCtx(name));
  auto protoName = formatv("{0}{1}", cppNamespace.namedType, name);
  if (addAsCase) addTo.addStandardCase(cppName, cppNamespace.namedType, name, deserializer);
  else addTo.addHelperMethod(formatv("deserialize{0}", protoName),
    {MethodParameter(protoName, varName)}, cppName, deserializer);
}

inline static void aggregateEnumAttr(const EnumAttr &def,
                                     const CppTypeInfo &cppNamespace,
                                     const std::string &varName,
                                     CppProtoDeserializer &addTo) {
  auto name = normalizeEnumName(def.getEnumClassName());
  auto cppName = name;
  if (name == "TLSModelAttr") {
    cppName = "TLS_ModelAttr";
  }
  cppName = formatv("{0}::{1}", cppNamespace.factualType, cppName).str();
  auto nameSnake = llvm::convertToSnakeFromCamelCase(name);

  const char *const enumDeserializerFmt = R"(
auto enumDeser = EnumDeserializer::deserialize{0}{1}({2}.value());
return {3}::get(&mInfo.ctx, enumDeser);)";

  auto deserializer = formatv(enumDeserializerFmt, cppNamespace.namedType, llvm::StringRef(name).drop_back(4), varName, cppName);

  addTo.addStandardCase(cppName, cppNamespace.namedType, name, deserializer);
}

static bool emitAttrProtoDeserializer(const RecordKeeper &records,
                                      raw_ostream &os,
                                      bool emitDecl) {
  const char *const defHeaderOpen = R"(
#include "cir-tac/AttrDeserializer.h"
#include "cir-tac/EnumDeserializer.h"
#include "proto/attr.pb.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;
using mlir::Attribute;
)";

  const char *const declHeaderOpen = R"(
#pragma once

#include "Util.h"
#include "proto/attr.pb.h"
#include "cir-tac/Deserializer.h"

#include <clang/CIR/Dialect/IR/CIRAttrs.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>

namespace protocir {
)";

  const char *const declHeaderClose = R"(
} // namespace protocir
)";

  const char *const namedAttrDeserializer = R"(
  auto nameDeser = deserializeMLIRStringAttr(pAttr.name());
  auto valueDeser = deserializeMLIRAttribute(pAttr.value());
  return mlir::NamedAttribute(nameDeser, valueDeser);)";

  const char *const flatSymbolDeserializer = R"(
auto rootReferenceDeser = deserializeMLIRStringAttr(pAttr.root_reference());
return mlir::FlatSymbolRefAttr::get(rootReferenceDeser);)";

  const char *const denseArrayDeserializer = R"(
std::vector<int> contentDeser;
auto sz = pAttr.raw_data_size();
for (int i = 0; i < sz; i++) {
  contentDeser.push_back(pAttr.raw_data(i));
}
return mlir::DenseI32ArrayAttr::get(&mInfo.ctx, contentDeser);)";

  const char *const langAttrDeserializer = R"(
auto langDeser = EnumDeserializer::deserializeCIRSourceLanguage(pAttr.lang());
auto lAttr = cir::SourceLanguageAttr::get(&mInfo.ctx, langDeser);
return cir::LangAttr::get(&mInfo.ctx, lAttr);
)";

  const char *const locationDeserializer = R"(
  switch (pAttr.location_case()) {
    case MLIRLocation::LocationCase::kCallSiteLoc: {
      return deserializeMLIRCallSiteLoc(pAttr.call_site_loc());
    } break;
    case MLIRLocation::LocationCase::kFileLineColLoc: {
      return deserializeMLIRFileLineColLoc(pAttr.file_line_col_loc());
    } break;
    case MLIRLocation::LocationCase::kFusedLoc: {
      return deserializeMLIRFusedLoc(pAttr.fused_loc());
    } break;
    case MLIRLocation::LocationCase::kNameLoc: {
      return deserializeMLIRNameLoc(pAttr.name_loc());
    } break;
    case MLIRLocation::LocationCase::kOpaqueLoc: {
      return deserializeMLIROpaqueLoc(pAttr.opaque_loc());
    } break;
    case MLIRLocation::LocationCase::kUnknownLoc: {
      return deserializeMLIRUnknownLoc(pAttr.unknown_loc());
    } break;
    default:
      llvm_unreachable("NYI");
      break;
  };)";

  auto mlirDefs = records.getAllDerivedDefinitionsIfDefined("Builtin_Attr");
  auto mlirLocationDefs =
      records.getAllDerivedDefinitionsIfDefined("Builtin_LocationAttr");
  auto cirDefs = records.getAllDerivedDefinitionsIfDefined("CIR_Attr");
  auto cirEnumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  const auto *varName = "pAttr";

  CppProtoDeserializer deserClass("AttrDeserializer", {"mlir::Attribute", "MLIRAttribute"},
    "MLIRAttribute", "Attribute", varName, declHeaderOpen, declHeaderClose, defHeaderOpen, "");

  deserClass.addField("ModuleInfo &", "mInfo");

  CppTypeInfo mlirNaming = {"mlir", "MLIR"};
  CppTypeInfo cirNaming = {"cir", "CIR"};

  for (auto *def : mlirDefs) {
    AttrDef attr(def);
    auto name = normalizeName(attr.getName());
    if (mlirAttributeWhitelist.count(name)) {
      aggregateAttribute(attr, mlirNaming, varName, deserClass);
    }
  }
  // mlir::NamedAttribute is not an instance of mlir::Attribute, but is used in various situations
  // adding its deserializer as a helper method so we have access to it, but is not actually present
  // in the MLIRAttribute switch function
  deserClass.addHelperMethod("deserializeMLIRNamedAttr", MethodParameter("MLIRNamedAttr", "pAttr"),
    "mlir::NamedAttribute", namedAttrDeserializer);
  deserClass.addStandardCase("mlir::FlatSymbolRefAttr", "MLIR", "FlatSymbolRefAttr", flatSymbolDeserializer);
  deserClass.addStandardCase("mlir::DenseI32ArrayAttr", "MLIR", "DenseI32ArrayAttr", denseArrayDeserializer);
  deserClass.addStandardCase("mlir::Location", "MLIR", "Location", locationDeserializer);
  deserClass.addStandardCase("cir::LangAttr", "CIR", "LangAttr", langAttrDeserializer);
  for (auto *def : mlirLocationDefs) {
    AttrDef attr(def);
    aggregateAttribute(attr, mlirNaming, varName, deserClass, false);
  }
  for (auto *def : cirDefs) {
    AttrDef attr(def);
    if (attr.getName().starts_with("AST")
        // a special case that needs one extra step before building;
        // deserializer cannot be generated automatically
        || attr.getName().ends_with("LangAttr")) {
      continue;
    }
    aggregateAttribute(attr, cirNaming, varName, deserClass);
  }
  for (auto *def : cirEnumDefs) {
    EnumAttr attr(def);
    aggregateEnumAttr(attr, cirNaming, varName, deserClass);
  }

  generateCodeFile({&deserClass}, /*disableClang=*/true, /*addLicense=*/false,
    emitDecl, os);

  return false;
}

static bool emitAttrProtoDeserializerSource(const RecordKeeper &records,
  llvm::raw_ostream &os) {
return emitAttrProtoDeserializer(records, os, /*emitDecl=*/false);
}

static bool emitAttrProtoDeserializerHeader(const RecordKeeper &records,
  llvm::raw_ostream &os) {
return emitAttrProtoDeserializer(records, os, /*emitDecl=*/true);
}

static mlir::GenRegistration
genAttrProto(
  "gen-attr-proto",
  "Generate proto file for attributes",
  &emitAttrProto);

static mlir::GenRegistration
genAttrProtoSerializerHeader(
  "gen-attr-proto-serializer-header",
  "Generate proto serializer .h for attributes",
  &emitAttrProtoSerializerHeader);

static mlir::GenRegistration
genAttrProtoSerializerSource(
  "gen-attr-proto-serializer-source",
  "Generate proto serializer .cpp for attributes",
  &emitAttrProtoSerializerSource);

static mlir::GenRegistration
genAttrProtoDeserializerHeader(
  "gen-attr-proto-deserializer-header",
  "Generate proto deserializer .h for attributes",
  &emitAttrProtoDeserializerHeader);

static mlir::GenRegistration
genAttrProtoDeserializerSource(
  "gen-attr-proto-deserializer-source",
  "Generate proto deserializer .cpp for attributes",
  &emitAttrProtoDeserializerSource);

static mlir::GenRegistration
getAttrKotlin(
  "gen-attr-kotlin",
  "Generate kotlin classes for attributes",
  &emitAttrKotlin);

static mlir::GenRegistration
getAttrKotlinBuilder(
  "gen-attr-kotlin-builder",
  "Generate kotlin builder for attributes",
  &emitAttrKotlinBuilder);
