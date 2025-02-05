#include "OpGenHelpers.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pass.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include "VespaGen.h"

#include <map>
#include <set>
#include <string>
#include <numeric>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

using namespace vespa;

static const char *const tblgenNamePrefix = "tblgen_";
static const char *const generatedArgName = "odsArg";
static const char *const odsBuilder = "odsBuilder";
static const char *const builderOpState = "odsState";
static const char *const propertyStorage = "propStorage";
static const char *const propertyValue = "propValue";
static const char *const propertyAttr = "propAttr";
static const char *const propertyDiag = "emitError";
static const char *const operandSegmentAttrName = "operandSegmentSizes";
static const char *const resultSegmentAttrName = "resultSegmentSizes";

const char *const deserializerFileHeader = R"(
#include "cir-tac/EnumsDeserializer.h"
#include "cir-tac/Deserializer.h"
#include "cir-tac/AttrDeserializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;
)";

const char *const deserializerDefStart = R"(
mlir::Operation Deserializer::deserializeOp(FunctionInfo &fInfo,
                                            const CIROp &pOp) {
  auto builder = fInfo.owner.builder;
  auto ctx = &fInfo.owner.ctx;
  auto module = &fInfo.owner;
  
  switch (pOp.operation_case()) {
)";

const char *const deserializerDefEnd = R"(
    case CIROp::OperationCase::OPERATION_NOT_SET:
      llvm_unreachable("Op kind not set!");
    default:
      llvm_unreachable("NYI");
  }
}
)";

const char *const deserializerCaseStart = R"(
    case CIROp::OperationCase::k{0}: {{
        auto p{0} = pOp.{1}();
)";

const char *const deserializerAttributeSet = R"(
        auto {1}Deser = Deserializer::deserialize{1}(p{0}.{2}());
)";

const char *const deserializerResultSet = R"(
        auto {1}Deser = Deserializer::getType(module, p{0}.{1}().id());
)";

const char *const deserializerOperandSet = R"(
        auto {1}Deser = Deserializer::getValue(fInfo, p{0}.{1}());
)";

const char *const deserializerCaseEnd = R"(
        return builder.create<cir::{0}>({1});
      } break;
)";

const char *const deserializerFileEnd = R"(
} // namespace protocir
)";

static bool emitOpProtoDeserializer(const RecordKeeper &records,
                                    raw_ostream &os) {
  os << autogenMessage;
  os << jacoDBLicense;
  os << deserializerFileHeader;

  os << deserializerDefStart;
  std::vector<const Record *> defs = getRequestedOpDefinitions(records);
  for (auto *def : defs) {
    Operator op(*def);
    assert(!op.skipDefaultBuilders() && "default builders are expected to be present!");

    auto opName = op.getCppClassName();
    
    std::vector<std::string> args;

    for (int i = 0; i < op.getNumResults(); i++) {
      std::string resultName = op.getResult(i).name.str();
      os << llvm::formatv(deserializerResultSet, opName, resultName);
      args.push_back(resultName + "Deser");
    }
    for (int i = 0; i < op.getNumOperands(); i++) {
      std::string operandName = op.getOperand(i).name.str();
      os << llvm::formatv(deserializerOperandSet, opName, operandName);
      args.push_back(operandName + "Deser");
    }
    for (int i = 0; i < op.getNumAttributes(); i++) {
      std::string attrName = op.getAttribute(i).name.str();
      std::string attrNameSnake = llvm::convertToSnakeFromCamelCase(attrName);
      os << llvm::formatv(deserializerAttributeSet, opName, attrName, attrNameSnake);
      args.push_back(attrName + "Deser");
    }

    auto argsCall = args.size() == 0 ? "" : std::accumulate(
      std::next(args.begin()), 
      args.end(), 
      args[0], 
      [](std::string a, std::string b) {
        return a + ", " + b;
      }
    );

    os << llvm::formatv(deserializerCaseEnd, opName, argsCall);
  }

  os << deserializerDefEnd;
  os << deserializerFileEnd;

  return false;
}

static mlir::GenRegistration
    genOpSerializerProto("gen-op-deser-proto",
                         "Generate op deserializer of Proto format",
                         &emitOpProtoDeserializer);
