#include "VespaCommon.h"

using namespace vespa;

void CppSource::printCodeBlock(raw_indented_ostream &os,
                               std::string code) {
  os.indent();
  os.printReindented(code);
  os.unindent();
}

void CppProtoSerializer::dumpSwitchFunc(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  os << formatv("{0} {1};\n", resTy.factualType, serName);
  if (preCaseBody) os.printReindented(preCaseBody.value());
  os << formatv("llvm::TypeSwitch<{0}>({1})\n", inputTy, inputName);
  for (auto c : cases) {
    os << formatv(".Case<{0}>([&]({0} {1}) {{\n", c.cppType, inputName);
    printCodeBlock(os, c.caseBody);
    os << "})\n";
  }
  os << formatv(".Default([]({0} {1}) {{\n"
                "  {1}.dump();\n"
                "  llvm_unreachable(\"unknown {1} during serialization\");\n"
                "});\n", inputTy, inputName);
  if (postCaseBody) os.printReindented(postCaseBody.value());
  os << formatv("return {0};", serName);
}

void CppProtoDeserializer::dumpSwitchFunc(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  if (preCaseBody) os.printReindented(preCaseBody.value());
  os << formatv("switch ({0}) {{\n", inputName);
  {
    raw_indented_ostream::DelimitedScope scope(os);
    for (auto c : cases) {
      os << formatv("case {0}: {{\n", c.caseValue);
      printCodeBlock(os, c.caseBody);
      os << "} break;\n";
    }
    os << "default:\n";
    os << "  llvm_unreachable(\"NYI\");\n";
    os << "  break;\n";
  }
  if (postCaseBody) os.printReindented(postCaseBody.value());
  os << "}\n";
}

void CppProtoSerializer::addCaseMethod(const SwitchCase &cas, std::string methodName) {
  auto &caseFuncBody =
    internalClass.addMethod(cas.protoType, methodName,
                            MethodParameter(cas.cppType, inputName))
    ->body();
  
  printCodeBlock(caseFuncBody.getStream(), cas.translatorBody);
}

void CppProtoDeserializer::addCaseMethod(const SwitchCase &cas, std::string methodName) {
  auto &caseFuncBody =
    internalClass.addMethod(cas.cppType, methodName,
                            MethodParameter(cas.protoType, inputName))
    ->body();
  
  printCodeBlock(caseFuncBody.getStream(), cas.translatorBody);
}

void CppSource::genClass() {
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

  auto &mainFuncBody =
    internalClass.addMethod(resTy.factualType, formatv("{0}{1}", funcName, resTy.namedType),
                            MethodParameter(inputTy, inputName))
    ->body();

  dumpSwitchFunc(mainFuncBody.getStream());

  for (auto cas : cases) {
    addCaseMethod(cas, formatv("{0}{1}", funcName, cas.protoType));
  }
}
