#include "VespaCommon.h"
#include "mlir/TableGen/Class.h"
#include "llvm/ADT/SmallVector.h"

using namespace vespa;

void CppSwitchSource::printCodeBlock(raw_indented_ostream &os,
                                     std::string code) {
  os.indent();
  os.printReindented(code);
  os.unindent();
}

void CppProtoSerializer::dumpSwitchFunc(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  os << formatv("{0} {1};\n", resTy.namedType, serName);
  if (preCaseBody) os.printReindented(preCaseBody.value());
  os << formatv("llvm::TypeSwitch<{0}>({1})\n", resTy.factualType, inputName);
  for (auto c : cases) {
    os << formatv(".Case<{0}>([&]({0} {1}) {{\n", c.cppType, inputName);
    printCodeBlock(os, c.caseBody);
    os << "})\n";
  }
  os << formatv(".Default([]({0} {1}) {{\n"
                "  {1}.dump();\n"
                "  llvm_unreachable(\"unknown {1} during serialization\");\n"
                "});\n", resTy.factualType, inputName);
  if (postCaseBody) os.printReindented(postCaseBody.value());
  os << formatv("return {0};", serName);
}

void CppProtoDeserializer::dumpSwitchFunc(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  if (preCaseBody) os.printReindented(preCaseBody.value());
  os << formatv("switch ({0}) {{\n", switchExpr);
  {
    raw_indented_ostream::DelimitedScope scope2(os);
    for (auto c : cases) {
      os << formatv("case {0}: {{\n", c.caseValue);
      printCodeBlock(os, c.caseBody);
      os << "} break;\n";
    }
    os << "default:\n";
    os << "  llvm_unreachable(\"NYI\");\n";
    os << "  break;\n";
  }
  os << "}\n";
  if (postCaseBody) os.printReindented(postCaseBody.value());
}

Method *CppProtoSerializer::addMethod(std::string methodName, std::string returnType,
                                      llvm::ArrayRef<MethodParameter> params) {
  return internalClass.addMethod(returnType, methodName, params);
}

Method *CppProtoSerializer::addTranslatorMethod(std::string protoType, std::string cppType,
                                                std::string methodName) {
  llvm::SmallVector<MethodParameter, 1> param{{cppType, inputName}};
  return addMethod(methodName, protoType, param);
}

Method *CppProtoDeserializer::addMethod(std::string methodName, std::string returnType,
                                        llvm::ArrayRef<MethodParameter> params) {
  std::vector<MethodParameter> staticParams = funcParams;
  for (const auto &param : params) {
    staticParams.emplace_back(param);
  }
  return internalClass.addMethod<Method::Static>(returnType, methodName, staticParams);
}

Method *CppProtoDeserializer::addTranslatorMethod(std::string protoType, std::string cppType,
                                                  std::string methodName) {
  llvm::SmallVector<MethodParameter, 1> param{{protoType, inputName}};
  return addMethod(methodName, cppType, param);
}

void CppSwitchSource::genCtr() {
  llvm::SmallVector<MethodParameter> ctrParams;
  for (auto field : fields) {
    if (field.isCtrParam)
      ctrParams.emplace_back(field.typ, field.name);
    internalClass.addField(field.typ, field.name);
  }

  auto *ctr = internalClass.addConstructor<Method::Inline>(ctrParams);
  for (auto field : fields) {
    ctr->addMemberInitializer(field.name, field.init);
  }
}

void CppSwitchSource::genClass() {
  if (!fields.empty())
    genCtr();

  auto &mainFuncBody =
    addTranslatorMethod(resTy.namedType, resTy.factualType, formatv("{0}{1}", funcName, resTy.namedType))
    ->body();

  dumpSwitchFunc(mainFuncBody.getStream());

  for (auto cas : cases) {
    auto *method = addTranslatorMethod(cas.protoType, cas.cppType, formatv("{0}{1}", funcName, cas.protoType));
    printCodeBlock(method->body().getStream(), cas.translatorBody);
  }
}
