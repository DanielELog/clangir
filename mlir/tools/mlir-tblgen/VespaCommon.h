// -*- C++ -*-
#ifndef MLIR_TOOLS_MLIRTBLGEN_VESPACOMMON_H_
#define MLIR_TOOLS_MLIRTBLGEN_VESPACOMMON_H_

#include "mlir/TableGen/Class.h"
#include "mlir/Support/IndentedOstream.h"

using mlir::raw_indented_ostream;
using namespace mlir::tblgen;

using llvm::formatv;

namespace vespa {

class CppProtoSerializer {
public:
  struct SwitchCase {
    std::string caseType;
    std::string caseTypeName;
    std::string caseBody;
    std::string translatorBody;
    std::string resType;
  };

  struct PrivateField {
    std::string typ;
    std::string name;
    std::string init;
    bool isCtrParam;
  };

  struct HeaderInfo {
    std::string open;
    std::string close;

    HeaderInfo(std::string open, std::string close)
      : open(open), close(close) {}
  };

private:
  std::string funcName;
  std::string className;
  std::vector<SwitchCase> cases;
  std::vector<PrivateField> fields;

  std::string resTy;
  std::string inputTy;

  std::string inputName;
  std::string serName;

  HeaderInfo declHeader;
  HeaderInfo defHeader;

  std::optional<std::string> preCaseBody;
  std::optional<std::string> postCaseBody;

  Class internalClass;

  void dumpSwitchFunc(raw_indented_ostream &os);

  void dumpSwitchFunc_des(raw_indented_ostream &os);

  void printCodeBlock(raw_indented_ostream &os, std::string code);

  void genClass();

  void addCase(SwitchCase c) {
    cases.push_back(c);
  }

  void addField(PrivateField f) {
    fields.push_back(f);
  }

public:
  CppProtoSerializer(std::string className, std::string ret,
    std::string inputTy, std::string inputName, std::string declHedOpen,
    std::string declHedClose, std::string defHedOpen, std::string defHedClose)
      : funcName("serialize"), className(className), resTy(ret),
        inputTy(inputTy), inputName(inputName),
        declHeader(declHedOpen, declHedClose),
        defHeader(defHedOpen, defHedClose), internalClass(className) {
      serName =
        llvm::convertToCamelFromSnakeCase(formatv("p_{0}", inputName).str());
    };

  void addCase(std::string typ, std::string typName, std::string body,
               std::string translator) {
    addCase({typ, typName, body, translator, typName});
  }

  void addField(std::string typ, std::string name, std::string init) {
    fields.push_back({typ, name, init, /*isCtrParam=*/false});
  }

  void addField(std::string typ, std::string name) {
    fields.push_back({typ, name, name, /*isCtrParam=*/true});
  }

  void addStandardCase(std::string typ, std::string typName,
                       std::string translator) {
    auto snakeTypName = llvm::convertToSnakeFromCamelCase(typName);
    auto caseBody = formatv("auto serialized = serialize{0}({1});\n"
                            "*{2}.mutable_{3}() = serialized;\n", typName,
                            inputName, serName, snakeTypName);
    addCase(typ, typName, caseBody, translator);
  }

  void addPreCaseBody(std::string body) {
    preCaseBody = body;
  }

  void addPostCaseBody(std::string body) {
    postCaseBody = body;
  }

  void dumpDecl(llvm::raw_ostream &os) {
    os << declHeader.open << "\n";
    genClass();
    internalClass.finalize();
    internalClass.writeDeclTo(os);
    os << declHeader.close << "\n";
  }

  void dumpDef(llvm::raw_ostream &os) {
    os << defHeader.open << "\n";
    genClass();
    internalClass.finalize();
    internalClass.writeDefTo(os);
    os << defHeader.close << "\n";
  }
};

} // namespace vespa

#endif // MLIR_TOOLS_MLIRTBLGEN_VESPACOMMON_H_
