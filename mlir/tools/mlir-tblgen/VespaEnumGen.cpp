#include "VespaGen.h"

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

#include <string>
#include <vector>

using llvm::formatv;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using mlir::tblgen::EnumAttr;

using namespace vespa;

static std::string normalizeName(llvm::StringRef name) {
  if (name == "TLS_Model") {
    return "TLSModel";
  }
  return name.str();
}

static bool emitEnumProto(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << protoHeader;
  os << "\n";

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    os << formatv("enum CIR{0} {{\n", enumName);
    size_t index = 0;
    for (auto &enumerant : enumAttr.getAllCases()) {
      auto enumerantName =
          llvm::convertToCamelFromSnakeCase(enumerant.getSymbol(), true);
      os << formatv("  {0}_{1} = {2};\n", enumName, enumerantName, index++);
    }
    os << "}\n\n";
  }

  os << "enum CIRRecordKind {\n";
  os << "  RecordKind_Class = 0;\n";
  os << "  RecordKind_Union = 1;\n";
  os << "  RecordKind_Struct = 2;\n";
  os << "}\n";
  os << "\n";
  os << "enum MLIRSignednessSemantics {\n";
  os << "  SignednessSemantics_Signless = 0;\n";
  os << "  SignednessSemantics_Signed = 1;\n";
  os << "  SignednessSemantics_Unsigned = 2;\n";
  os << "}\n";
  os << "\n";

  os << clangOn;
  return false;
}

static std::string getEnumCppName(const EnumAttr &enumAttr) {
  auto enumName = enumAttr.getEnumClassName();

  // Inconsistency
  const auto *sob = enumName == "SignedOverflowBehavior" ? "sob::" : "";

  return formatv("cir::{0}{1}", sob, enumName);
}

static bool emitEnumProtoSerializerHeader(const RecordKeeper &records,
                                          raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;

  os << "\n";
  os << "#pragma once\n";
  os << "\n";
  os << "#include \"proto/enum.pb.h\"\n";
  os << "\n";
  os << "#include <clang/CIR/Dialect/IR/CIROpsEnums.h>\n";
  os << "#include <clang/CIR/Dialect/IR/CIRTypes.h>\n";
  os << "#include <mlir/IR/BuiltinTypes.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    // Inconsistency
    const auto *sob = enumName == "SignedOverflowBehavior" ? "sob::" : "";
    os << formatv("CIR{0} serializeCIR{0}(cir::{2}{1});\n", enumName,
                  enumAttr.getEnumClassName(), sob);
  }

  os << formatv(
                "CIRRecordKind serializeCIRRecordKind(cir::StructType::RecordKind);\n");
  os << "MLIRSignednessSemantics serializeMLIRSignednessSemantics(mlir::IntegerType::SignednessSemantics);\n";
  os << "\n";
  os << clangOn;

  return false;
}

static bool emitEnumProtoDeserializerHeader(const RecordKeeper &records,
                                            raw_ostream &os) {
  const char *const declHeaderOpen = R"(
#pragma once

#include "proto/enum.pb.h"

#include <clang/CIR/Dialect/IR/CIROpsEnums.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace protocir {

class EnumDeserializer {
public:)";

  const char *const declHeaderClose = R"(
};
} // namespace protocir
)";

  const char *const methodDecl = R"(
static {1} deserialize{2}{0}({2}{0});)";

  auto recordKindDecl = formatv(methodDecl, "RecordKind", "cir::StructType::RecordKind", "CIR");
  auto signednessDecl = formatv(methodDecl, "SignednessSemantics", "mlir::IntegerType::SignednessSemantics", "MLIR");

  os << autogenMessage;
  os << clangOff;
  os << declHeaderOpen;

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    auto enumCppName = getEnumCppName(enumAttr);
    os << formatv(methodDecl, enumName, enumCppName, "CIR");
  }

  os << recordKindDecl;
  os << signednessDecl;

  os << declHeaderClose;
  os << clangOn;

  return false;
}

static bool emitEnumProtoSerializerSource(const RecordKeeper &records,
                                          raw_ostream &os) {

  os << autogenMessage;
  os << clangOff;

  os << "\n";
  os << "#include \"cir-tac/EnumSerializer.h\"\n";
  os << "#include <clang/CIR/Dialect/IR/CIRTypes.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    // Inconsistency
    const auto *sob = enumName == "SignedOverflowBehavior" ? "sob::" : "";
    os << formatv("CIR{0} serializeCIR{0}(cir::{2}{1} e) {{\n", enumName,
                  enumAttr.getEnumClassName(), sob);
    os << "  switch (e) {\n";
    for (auto &enumerant : enumAttr.getAllCases()) {
      auto upperCamelEnumName =
          llvm::convertToCamelFromSnakeCase(enumerant.getSymbol(), true);
      os << formatv("  case cir::{2}{0}::{1}:\n", enumAttr.getEnumClassName(),
                    enumerant.getSymbol(), sob);
      os << formatv("    return CIR{0}::{0}_{1};\n", enumName,
                    upperCamelEnumName);
    }
    os << "  default:\n";
    os << "    assert(0 && \"Unknown enum variant\");\n";
    os << "  }\n";
    os << "}\n";
    os << "\n";
  }

  os << "CIRRecordKind serializeCIRRecordKind(cir::StructType::RecordKind e) "
        "{\n";
  os << "  switch (e) {\n";
  os << "  case cir::StructType::RecordKind::Class:\n";
  os << "    return CIRRecordKind::RecordKind_Class;\n";
  os << "  case cir::StructType::RecordKind::Union:\n";
  os << "    return CIRRecordKind::RecordKind_Union;\n";
  os << "  case cir::StructType::RecordKind::Struct:\n";
  os << "    return CIRRecordKind::RecordKind_Struct;\n";
  os << "  default:\n";
  os << "    assert(0 && \"Unknown enum variant\");\n";
  os << "  }\n";
  os << "}\n";
  os << "\n";

  os << "MLIRSignednessSemantics serializeMLIRSignednessSemantics(mlir::IntegerType::SignednessSemantics e) {\n";
  os << "  switch (e) {\n";
  os << "  case mlir::IntegerType::SignednessSemantics::Signless:\n";
  os << "    return MLIRSignednessSemantics::SignednessSemantics_Signless;\n";
  os << "  case mlir::IntegerType::SignednessSemantics::Signed:\n";
  os << "    return MLIRSignednessSemantics::SignednessSemantics_Signed;\n";
  os << "  case mlir::IntegerType::SignednessSemantics::Unsigned:\n";
  os << "    return MLIRSignednessSemantics::SignednessSemantics_Unsigned;\n";
  os << "  default:\n";
  os << "    assert(0 && \"Unknown enum variant\");\n";
  os << "  }\n";
  os << "}\n";

  os << "\n";
  os << clangOn;

  return false;
}

const char *const defHeader = R"(
#include "cir-tac/EnumDeserializer.h"
#include <clang/CIR/Dialect/IR/CIRTypes.h>

using namespace protocir;
)";

const char *const methodDef = R"(
{1} EnumDeserializer::deserialize{2}{0}({2}{0} e) {{
  switch (e) {{)";

const char *const switchCaseDef = R"(
    case {4}{0}::{0}_{1}:
      return {2}::{3};)";

const char *const switchEndDef = R"(
    default:
      llvm_unreachable("Unexpected enum value!");
  }
})";

struct EnumPairing {
  std::string cppName;
  std::string protoName;
};

static void emitEnumDeserializerMethod(const EnumPairing &name,
                                       const llvm::ArrayRef<EnumPairing> &values,
                                       const char *protoPrefix,
                                       raw_ostream &os) {
  os << formatv(methodDef, name.protoName, name.cppName, protoPrefix);
  for (auto enumerant : values) {
    os << formatv(switchCaseDef, name.protoName, enumerant.protoName, name.cppName,
      enumerant.cppName, protoPrefix);
  }
  os << switchEndDef;
}

static bool emitEnumProtoDeserializerSource(const RecordKeeper &records,
                                            raw_ostream &os) {
  const EnumPairing recordKindName = {"cir::StructType::RecordKind", "RecordKind"};

  const std::vector<EnumPairing> recordKindValues = {
    {"Class", "Class"},
    {"Union", "Union"},
    {"Struct", "Struct"},
  };

  const EnumPairing signedlessName = {"mlir::IntegerType::SignednessSemantics", "SignednessSemantics"};

  const std::vector<EnumPairing> signedlessValues = {
    {"Signless", "Signless"},
    {"Signed", "Signed"},
    {"Unsigned", "Unsigned"},
  };

  os << autogenMessage;
  os << clangOff;
  os << defHeader;

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    auto enumCppName = getEnumCppName(enumAttr);
    std::vector<EnumPairing> enumerants;
    for (auto &enumerant : enumAttr.getAllCases()) {
      auto enumerantName =
          llvm::convertToCamelFromSnakeCase(enumerant.getSymbol(), true);
      enumerants.push_back({enumerant.getSymbol().str(), enumerantName});
    }
    emitEnumDeserializerMethod({enumCppName, enumName}, enumerants, "CIR", os);
  }

  emitEnumDeserializerMethod(recordKindName, recordKindValues, "CIR", os);
  emitEnumDeserializerMethod(signedlessName, signedlessValues, "MLIR", os);

  os << "\n";
  os << clangOn;

  return false;
}

static bool emitEnumKotlin(const RecordKeeper &records, raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.api.cir.cfg\n";
  os << "\n";

  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    os << formatv("enum class CIR{0} {{\n", enumName);
    for (const auto &enumerant : enumAttr.getAllCases()) {
      auto upperCamelEnumName =
          llvm::convertToCamelFromSnakeCase(enumerant.getSymbol(), true);
      os << formatv("    {0},\n", upperCamelEnumName);
    }
    os << formatv("}\n");
    os << formatv("\n");
  }

  os << "enum class CIRRecordKind {\n";
  os << "    Class,\n";
  os << "    Union,\n";
  os << "    Struct,\n";
  os << "}\n";
  os << "\n";

  os << "enum class MLIRSignednessSemantics {\n";
  os << "    Signless,\n";
  os << "    Signed,\n";
  os << "    Unsigned,\n";
  os << "}\n";

  return false;
}

static bool emitEnumKotlinBuilder(const RecordKeeper &records,
                                  raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.impl.cfg.builder\n";
  os << "\n";
  os << "import org.jacodb.api.cir.cfg.*\n";
  os << "import org.jacodb.impl.grpc.Enum\n";
  os << "\n";


  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (auto &def : defs) {
    EnumAttr enumAttr(def);
    auto enumName = normalizeName(enumAttr.getEnumClassName());
    os << formatv("fun buildCIR{0}(enum : Enum.CIR{0}) = when (enum) {{\n",
                  enumName);
    for (const auto &enumerant : enumAttr.getAllCases()) {
      auto upperCamelEnumName =
          llvm::convertToCamelFromSnakeCase(enumerant.getSymbol(), true);
      os << formatv("    Enum.CIR{0}.{0}_{1} -> CIR{0}.{1}\n", enumName,
                    upperCamelEnumName);
    }
    os << formatv("    Enum.CIR{0}.UNRECOGNIZED -> throw Exception()\n",
                  enumName);
    os << formatv("}\n");
    os << formatv("\n");
  }

  os << "fun buildCIRRecordKind(enum : Enum.CIRRecordKind) = when (enum) "
        "{\n";
  os << "    Enum.CIRRecordKind.CIRRecordKind_Class -> "
        "CIRRecordKind.Class\n";
  os << "    Enum.CIRRecordKind.CIRRecordKind_Union -> "
        "CIRRecordKind.Union\n";
  os << "    Enum.CIRRecordKind.CIRRecordKind_Struct -> "
        "CIRRecordKind.Struct\n";
  os << "    Enum.CIRRecordKind.UNRECOGNIZED -> throw Exception()\n";
  os << "}\n";
  os << "\n";

  os << "fun buildMLIRSignednessSemantics(enum : Enum.MLIRSignednessSemantics) = when (enum) "
        "{\n";
  os << "    Enum.MLIRSignednessSemantics.MLIRSignednessSemantics_Signless -> "
        "MLIRSignednessSemantics.Signless\n";
  os << "    Enum.MLIRSignednessSemantics.MLIRSignednessSemantics_Signed -> "
        "MLIRSignednessSemantics.Signed\n";
  os << "    Enum.MLIRSignednessSemantics.MLIRSignednessSemantics_Unsigned -> "
        "MLIRSignednessSemantics.Unsigned\n";
  os << "    Enum.MLIRSignednessSemantics.UNRECOGNIZED -> throw Exception()\n";
  os << "}\n";

  return false;
}

static mlir::GenRegistration genEnumProto("gen-enum-proto",
                                          "Generate proto file for enums",
                                          &emitEnumProto);

static mlir::GenRegistration
    genEnumProtoSerializerHeader("gen-enum-proto-serializer-header",
                                 "Generate proto serializer .h for enums",
                                 &emitEnumProtoSerializerHeader);

static mlir::GenRegistration
    genEnumProtoSerializerSource("gen-enum-proto-serializer-source",
                                 "Generate proto serializer .cpp for enums",
                                 &emitEnumProtoSerializerSource);

static mlir::GenRegistration genEnumKotlin("gen-enum-kotlin",
                                           "Generate kotlin for enums",
                                           &emitEnumKotlin);

static mlir::GenRegistration
    genEnumKotlinBuilder("gen-enum-kotlin-builder",
                         "Generate kotlin builder for enums",
                         &emitEnumKotlinBuilder);

static mlir::GenRegistration
genEnumProtoDeserializerHeader(
  "gen-enum-proto-deserializer-header",
  "Generate proto deserializer .h for enums",
  &emitEnumProtoDeserializerHeader);

static mlir::GenRegistration
genEnumProtoDeserializerSource(
  "gen-enum-proto-deserializer-source",
  "Generate proto deserializer .cpp for enums",
  &emitEnumProtoDeserializerSource);
