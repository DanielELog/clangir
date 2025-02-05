#include "VespaGen.h"

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using llvm::formatv;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::AttrOrTypeDef;

using namespace vespa;

static std::string normalizeTypeName(StringRef name) {
  if (name.ends_with("Type")) {
    name = name.drop_back(4);
  }
  if (name.starts_with("Builtin_")) {
    return formatv("MLIR{0}Type", name.drop_front(8));
  }
  if (name.starts_with("CIR_")) {
    return formatv("CIR{0}Type", name.drop_front(4));
  }
  assert(0 && "Unknown kind of type name");
  return name.str();
}

static std::string getParamVarName(StringRef param) {
  return param.str() + "Deser";
}

static std::string getCppType(AttrOrTypeDef &type) {
  auto dialect = type.getDialect();
  if (dialect.getName() == "builtin") {
    return formatv("mlir::{0}", type.getCppClassName());
  }
  if (dialect.getName() == "cir") {
    return formatv("cir::{0}", type.getCppClassName());
  }
  assert(0 && "Unknown dialect");
  return "";
}

static std::vector<const Record *> getDefs(const RecordKeeper &records) {
  std::vector<const Record *> mlir =
      records.getAllDerivedDefinitionsIfDefined("Builtin_Type");
  auto cir = records.getAllDerivedDefinitionsIfDefined("CIR_Type");
  for (auto &record : cir) {
    mlir.push_back(record);
  }
  return mlir;
}

static bool emitTypeProto(const RecordKeeper &records, llvm::raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << jacoDBLicense;
  os << protoHeader;
  os << "\n";
  os << "import \"setup.proto\";\n";
  os << "import \"enum.proto\";\n";
  os << "import \"attr.proto\";\n";
  os << "\n";

  auto defs = getDefs(records);

  os << "message MLIRType {\n";
  os << "  MLIRTypeID id = 1;\n";
  os << "  oneof type {\n";

  size_t index = 2;
  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    auto snake = llvm::convertToSnakeFromCamelCase(name);
    os << formatv("    {0} {1} = {2};\n", name, snake, index++);
  }

  os << formatv("    CIRStructType cir_struct_type = {0};\n", index++);

  os << "  }\n";
  os << "}\n";

  os << "\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    os << formatv("message {0} {{\n", name);
    size_t index = 1;
    for (auto param : type.getParameters()) {
      auto paramSnake = llvm::convertToSnakeFromCamelCase(param.getName());
      if (param.isOptional()) {
        os << formatv("  optional {0} {1} = {2};\n", getProtoType(param), paramSnake,
                      index++);
      } else {
        os << formatv("  {0} {1} = {2};\n", getProtoType(param), paramSnake,
                      index++);
      }
    }
    os << "}\n";
    os << "\n";
  }

  os << "message CIRStructType {\n";
  os << "  repeated MLIRTypeID members = 1;\n";
  os << "  MLIRStringAttr name = 2;\n";
  os << "  bool incomplete = 3;\n";
  os << "  bool packed = 4;\n";
  os << "  CIRRecordKind kind = 5;\n";
  os << "}\n";
  os << "\n";

  os << clangOn;
  return false;
}

static bool emitTypeProtoSerializerHeader(const RecordKeeper &records,
                                          llvm::raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << jacoDBLicense;
  os << "\n";

  auto defs = getDefs(records);

  os << "#pragma once\n";
  os << "\n";
  os << "#include \"Util.h\"\n";
  os << "#include \"cir-tac/AttrSerializer.h\"\n";
  os << "#include \"proto/setup.pb.h\"\n";
  os << "#include \"proto/type.pb.h\"\n";
  os << "\n";
  os << "#include <clang/CIR/Dialect/IR/CIRTypes.h>\n";
  os << "#include <mlir/IR/BuiltinTypes.h>\n";
  os << "#include <mlir/IR/Builders.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  os << "class TypeSerializer {\n";
  os << "public:\n";
  os << "  TypeSerializer(MLIRModuleID moduleID, TypeCache &typeCache)\n";
  os << "  : moduleID(moduleID), typeCache(typeCache), attributeSerializer(moduleID, typeCache) {}\n";
  os << "\n";
  os << "  MLIRType serializeMLIRType(mlir::Type);\n";
  os << "\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    os << formatv("  {0} serialize{0}({1});\n", name,
                  getCppType(type));
  }

  os << "  CIRStructType serializeCIRStructType(cir::StructType);\n";

  os << "\n";

  os << "private:\n";
  os << "  MLIRModuleID moduleID;\n";
  os << "  TypeCache &typeCache;\n";
  os << "  AttributeSerializer attributeSerializer;\n";
  os << "};\n";
  os << "\n";

  os << clangOn;
  return false;
}

static bool emitTypeProtoSerializerSource(const RecordKeeper &records,
                                          llvm::raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << "\n";

  os << "#include \"cir-tac/TypeSerializer.h\"\n";
  os << "#include \"cir-tac/EnumSerializer.h\"\n";
  os << "#include \"proto/type.pb.h\"\n";
  os << "\n";
  os << "#include <llvm/ADT/TypeSwitch.h>\n";
  os << "#include <mlir/IR/BuiltinTypes.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = getDefs(records);

  os << "MLIRType TypeSerializer::serializeMLIRType(mlir::Type type) {\n";
  os << "  MLIRType pType;\n";
  os << "\n";
  os << "  *pType.mutable_id() = typeCache.getMLIRTypeID(type);\n";
  os << "\n";
  os << "  llvm::TypeSwitch<mlir::Type>(type)\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    auto snake = llvm::convertToSnakeFromCamelCase(name);

    os << formatv("  .Case<{0}>([this, &pType]({0} type) {{\n",
                  getCppType(type));
    os << formatv("    auto serialized = serialize{0}(type);\n", name);
    os << formatv("    *pType.mutable_{0}() = serialized;\n", snake);
    os << formatv("  })\n");
  }

  os << "  .Case<cir::StructType>([this, &pType](cir::StructType type) {\n";
  os << "    auto serialized = serializeCIRStructType(type);\n";
  os << "    *pType.mutable_cir_struct_type() = serialized;\n";
  os << "  })\n";

  os << "  .Default([](mlir::Type type) {\n";
  os << "    type.dump();\n";
  os << "    llvm_unreachable(\"unknown type during serialization\");\n";
  os << "  });\n";
  os << "\n";
  os << "  return pType;\n";
  os << "}\n";

  os << "\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());

    os << formatv("{0} TypeSerializer::serialize{0}({1} type) {{\n", name,
                  getCppType(type));
    os << formatv("  {0} serialized;\n", name);
    for (auto param : type.getParameters()) {
      serializeParameter(param, "type", os);
    }
    os << "  return serialized;\n";
    os << "}\n";
    os << "\n";
  }

  os << "CIRStructType TypeSerializer::serializeCIRStructType(cir::StructType "
        "type) {\n";
  os << "  CIRStructType serialized;\n";
  os << "  for (auto i : type.getMembers()) {\n";
  os << "    serialized.mutable_members()->Add(typeCache.getMLIRTypeID(i));\n";
  os << "  }\n";
  os << "  *serialized.mutable_name() = attributeSerializer.serializeMLIRStringAttr(type.getName());\n";
  os << "  serialized.set_incomplete(type.getIncomplete());\n";
  os << "  serialized.set_packed(type.getPacked());\n";
  os << "  serialized.set_kind(serializeCIRRecordKind(type.getKind()));\n";
  os << "  return serialized;\n";
  os << "}\n";
  os << "\n";

  os << clangOn;
  return false;
}

static bool emitTypeProtoSerializer(const RecordKeeper &records,
                                    llvm::raw_ostream &os,
                                    bool emitDecl) {
  std::string defHeader = R"(
#include "cir-tac/TypeSerializer.h"
#include "cir-tac/EnumSerializer.h"
#include "proto/type.pb.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace protocir;
)";

  std::string declHeader = R"(
#pragma once

#include "Util.h"
#include "cir-tac/AttrSerializer.h"
#include "proto/setup.pb.h"
#include "proto/type.pb.h"

#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>

using namespace protocir;
)";

  std::string structSer = R"(
CIRStructType serialized;
for (auto i : type.getMembers()) {
  serialized.mutable_members()->Add(typeCache.getMLIRTypeID(i));
}
*serialized.mutable_name() = attributeSerializer.serializeMLIRStringAttr(type.getName());
serialized.set_incomplete(type.getIncomplete());
serialized.set_packed(type.getPacked());
serialized.set_kind(serializeCIRRecordKind(type.getKind()));
return serialized;
)";

  auto defs = getDefs(records);

  auto serClass = CppProtoSerializer("TypeSerializer", "MLIRType", "mlir::Type",
    "type", declHeader, defHeader);

  serClass.addField("MLIRModuleID", "moduleID");
  serClass.addField("TypeCache &", "typeCache");
  serClass.addField("AttributeSerializer", "attributeSerializer",
                    "moduleID, typeCache");

  serClass.addPreCaseBody("*pType.mutable_id() = typeCache.getMLIRTypeID(type);\n");

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    auto serializer = serializeParameters(name, type.getParameters(), "type");
    serClass.addStandardCase(getCppType(type), name, serializer);
  }
  serClass.addStandardCase("cir::StructType", "CIRStructType", structSer);

  generateCodeFile(serClass, /*disableClang=*/true, /*addLicense=*/false,
                   emitDecl, os);

  return false;
}

static bool emitTypeProtoSerializerSourceTest(const RecordKeeper &records,
                                                llvm::raw_ostream &os) {
  return emitTypeProtoSerializer(records, os, /*emitDecl=*/false);
}

static bool emitTypeProtoSerializerHeaderTest(const RecordKeeper &records,
                                                llvm::raw_ostream &os) {
  return emitTypeProtoSerializer(records, os, /*emitDecl=*/true);
}

static bool emitTypeProtoDeserializerSource(const RecordKeeper &records,
                                            llvm::raw_ostream &os) {
  os << autogenMessage;
  os << clangOff;
  os << "\n";

  os << "#include \"cir-tac/TypeDeserializer.h\"\n";
  os << "#include \"cir-tac/EnumDeserializer.h\"\n";
  os << "#include \"proto/type.pb.h\"\n";
  os << "\n";
  os << "#include <llvm/ADT/TypeSwitch.h>\n";
  os << "#include <mlir/IR/BuiltinTypes.h>\n";
  os << "\n";
  os << "using namespace protocir;\n";
  os << "\n";

  auto defs = getDefs(records);

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());

    os << formatv("{1} TypeDeserializer::deserialize{0}({0} pType) {{\n", name,
                  getCppType(type));
    for (auto param : type.getParameters()) {
      auto paramName = getParamVarName(param.getName());
      deserializeParameter(param, "pType", os);
    }
    os << "  return serialized;\n";
    os << "}\n";
    os << "\n";
  }

  os << "CIRStructType TypeSerializer::serializeCIRStructType(cir::StructType "
        "type) {\n";
  os << "  CIRStructType serialized;\n";
  os << "  for (auto i : type.getMembers()) {\n";
  os << "    serialized.mutable_members()->Add(typeCache.getMLIRTypeID(i));\n";
  os << "  }\n";
  os << "  *serialized.mutable_name() = attributeSerializer.serializeMLIRStringAttr(type.getName());\n";
  os << "  serialized.set_incomplete(type.getIncomplete());\n";
  os << "  serialized.set_packed(type.getPacked());\n";
  os << "  serialized.set_kind(serializeCIRRecordKind(type.getKind()));\n";
  os << "  return serialized;\n";
  os << "}\n";
  os << "\n";

  os << clangOn;
  return false;
}

static bool emitTypeKotlin(const RecordKeeper &records, llvm::raw_ostream &os) {
  os << autogenMessage;
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.api.cir.cfg\n";
  os << "\n";

  auto defs = getDefs(records);

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    os << formatv("data class {0}(\n", name);
    os << formatv("    override val id: MLIRTypeID,\n");
    for (auto param : type.getParameters()) {
      auto paramCamel = llvm::convertToCamelFromSnakeCase(param.getName());
      if (param.isOptional()) {
        os << formatv("    val {0}: {1}?,\n", paramCamel, getKotlinType(param));
      } else {
        os << formatv("    val {0}: {1},\n", paramCamel, getKotlinType(param));
      }
    }
    os << ") : MLIRType\n";
    os << "\n";
  }

  os << "data class CIRStructType(\n";
  os << "    override val id: MLIRTypeID,\n";
  os << "    val members : List<MLIRTypeID>,\n";
  os << "    val name : MLIRStringAttr,\n";
  os << "    val incomplete : Boolean,\n";
  os << "    val packed : Boolean,\n";
  os << "    val kind: CIRRecordKind\n";
  os << ") : MLIRType\n";

  return false;
}

static bool emitTypeKotlinBuilder(const RecordKeeper &records,
                                  llvm::raw_ostream &os) {
  os << autogenMessage;
  os << "\n";
  os << jacoDBLicense;
  os << "\n";
  os << "package org.jacodb.impl.cfg.builder\n";
  os << "\n";
  os << "import org.jacodb.api.cir.cfg.*\n";
  os << "import org.jacodb.impl.grpc.Type\n";
  os << "\n";

  auto defs = records.getAllDerivedDefinitionsIfDefined("TypeDef");

  os << "fun buildMLIRType(type: Type.MLIRType): MLIRType {\n";
  os << "    val id = buildMLIRTypeID(type.id)\n";
  os << "    val builtType: MLIRType = when (type.typeCase!!) {\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    auto snake =
        llvm::convertToSnakeFromCamelCase(StringRef(name));
    auto lowerCamel = llvm::convertToCamelFromSnakeCase(snake, false);
    std::transform(snake.begin(), snake.end(), snake.begin(), ::toupper);
    os << formatv("        Type.MLIRType.TypeCase.{0} -> build{1}(id, type.{2})\n", snake,
                  name, lowerCamel);
  }
  os << formatv("        Type.MLIRType.TypeCase.CIR_STRUCT_TYPE -> buildCIRStructType(id, "
                "type.cirStructType)\n");
  os << formatv("        Type.MLIRType.TypeCase.TYPE_NOT_SET -> throw Exception()\n");
  os << formatv("    }\n");
  os << formatv("    return builtType\n");
  os << formatv("}\n");

  os << "\n";

  for (auto &def : defs) {
    AttrOrTypeDef type(def);
    auto name = normalizeTypeName(type.getName());
    os << formatv("fun build{0}(id: MLIRTypeID, type: Type.{0}) =\n",
                  name);
    os << formatv("    {0}(\n", name);
    os << formatv("        id,\n");
    for (auto param : type.getParameters()) {
        buildParameter(param, "type", os);
    }
    os << "    )\n";
    os << "\n";
  }

  os << "fun buildCIRStructType(id: MLIRTypeID, type: "
        "Type.CIRStructType) =\n";
  os << "    CIRStructType(\n";
  os << "        id,\n";
  os << "        buildMLIRTypeIDArray(type.membersList),\n";
  os << "        buildMLIRStringAttr(type.name),\n";
  os << "        type.incomplete,\n";
  os << "        type.packed,\n";
  os << "        buildCIRRecordKind(type.kind),\n";
  os << "    )\n";

  return false;
}

static mlir::GenRegistration
genTypeProto("gen-type-proto",
             "Generate proto file for types",
             &emitTypeProto);

static mlir::GenRegistration
genTypeProtoSerializerHeader("gen-type-proto-serializer-header",
                             "Generate proto serializer .h for types",
                             &emitTypeProtoSerializerHeader);

static mlir::GenRegistration
genTypeProtoSerializerSource("gen-type-proto-serializer-source",
                             "Generate proto serializer .cpp for types",
                             &emitTypeProtoSerializerSource);

static mlir::GenRegistration
genTypeKotlin("gen-type-kotlin",
              "Generate kotlin for types",
              &emitTypeKotlin);

static mlir::GenRegistration
genTypeKotlinBuilder("gen-type-kotlin-builder",
                     "Generate kotlin builder for types",
                     &emitTypeKotlinBuilder);

static mlir::GenRegistration
genTypeProtoSerializerSourceTest("gen-type-proto-serializer-source-test",
                                 "Generate proto serializer .cpp for types",
                                 &emitTypeProtoSerializerSourceTest);

static mlir::GenRegistration
genTypeProtoSerializerHeaderTest("gen-type-proto-serializer-header-test",
                                 "Generate proto serializer .h for types",
                                 &emitTypeProtoSerializerHeaderTest);
