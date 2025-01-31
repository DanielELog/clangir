#include "FormatGen.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "VespaGen.h"

using llvm::formatv;
using llvm::Record;
using llvm::RecordKeeper;
using namespace mlir;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::EnumAttrCase;

using namespace vespa;

const char *const deserializerDefFileHeader = R"(
#include "cir-tac/EnumsDeserializer.h"

namespace protocir {
)";

const char *const deserializerDeclFileHeader = R"(
#include "proto/enumgen.pb.h"

#include <clang/CIR/Dialect/IR/CIRDialect.h>
)";

const char *const deserializerDeclStart = R"(
namespace protocir {
class EnumsDeserializer {
public:
)";

const char *const deserializerDeclEnum = R"(
  static {1}
  deserialize{0}(const CIR{0} &pKind);
)";

const char *const deserializerDeclEnd = R"(
};
} // namespace protocir
)";

const char *const deserializerDefEnumStart = R"(
{1}
EnumsDeserializer::deserialize{0}(const CIR{0} &pKind) {{
  switch (pKind) {{)";

const char *const deserializerDefEnumCase = R"(
    case protocir::CIR{0}::{3}:
      return {1}::{2};
)";

const char *const deserializerDefEnumEnd = R"(
    default:
      llvm_unreachable("NYI");
  }
}
)";

const char *const deserializerDefEnd = R"(
} // namespace: protocir
)";

static void
emitEnumProtoDeserializerDef(StringRef enumName, StringRef fullEnumName,
                             const std::vector<EnumAttrCase> &enumerants,
                             raw_ostream &os) {
  os << formatv(deserializerDefEnumStart, enumName, fullEnumName);

  for (const auto &enumerant : enumerants) {
    auto symbol = makeIdentifier(enumerant.getSymbol());
    auto protoSymbol = makeProtoSymbol(symbol);
    os << formatv(deserializerDefEnumCase, enumName, fullEnumName, symbol,
                  makeFullProtoSymbol(enumName, protoSymbol));
  }
  os << formatv(deserializerDefEnumEnd);
}

static void emitEnumProtoDeserializerDecl(StringRef enumName,
                                          StringRef fullEnumName,
                                          StringRef description,
                                          raw_ostream &os) {
  os << "// " << description;
  os << formatv(deserializerDeclEnum, enumName, fullEnumName);
  os << "\n";
}

static void emitEnumProtoSerializer(const Record &enumDef, raw_ostream &os,
                                    bool emitDecl) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef namespaceName = enumAttr.getCppNamespace();
  StringRef description = enumAttr.getSummary();
  auto enumerants = enumAttr.getAllCases();

  if (emitDecl) {
    // Emit the enum serializer declarations
    emitEnumProtoDeserializerDecl(
        enumName, formatv("{0}::{1}", namespaceName, enumName).str(),
        description, os);
  } else {
    // Emit the enum serializer definition
    emitEnumProtoDeserializerDef(
        enumName, formatv("{0}::{1}", namespaceName, enumName).str(),
        enumerants, os);
  }
}

static bool emitEnumProtoDeserializerDecls(const RecordKeeper &records,
                                           raw_ostream &os) {
  os << autogenMessage;
  os << jacoDBLicense;
  os << deserializerDeclFileHeader;
  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  os << deserializerDeclStart;
  for (const Record *def : defs)
    emitEnumProtoSerializer(*def, os, /*emitDecl=*/true);
  os << deserializerDeclEnd;

  return false;
}

static bool emitEnumProtoDeserializerDefs(const RecordKeeper &records,
                                          raw_ostream &os) {
  os << autogenMessage;
  os << jacoDBLicense;
  os << deserializerDefFileHeader;
  auto defs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  for (const Record *def : defs)
    emitEnumProtoSerializer(*def, os, /*emitDecl=*/false);
  os << deserializerDefEnd;

  return false;
}

static mlir::GenRegistration
  genEnumDeserializerProtoDecls("gen-enum-deser-proto-decls",
                                "Generate enum Proto deserializer declarations",
                                &emitEnumProtoDeserializerDecls);

static mlir::GenRegistration
  genEnumDeserializerProtoDefs("gen-enum-deser-proto-defs",
                               "Generate enum Proto deserializer definitions",
                               &emitEnumProtoDeserializerDefs);
