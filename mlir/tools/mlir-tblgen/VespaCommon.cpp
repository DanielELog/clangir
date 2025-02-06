#include "VespaCommon.h"

using namespace vespa;

void CppProtoSerializer::printCodeBlock(raw_indented_ostream &os,
                                        std::string code) {
  os.indent();
  os.printReindented(code);
  os.unindent();
}

void CppProtoSerializer::dumpSwitchFunc(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  os << formatv("{0} {1};\n", resTy, serName);
  if (preCaseBody) os.printReindented(preCaseBody.value());
  os << formatv("llvm::TypeSwitch<{0}>({1})\n", inputTy, inputName);
  for (auto c : cases) {
    os << formatv(".Case<{0}>([&]({0} {1}) {{\n", c.caseType, inputName);
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

void CppProtoSerializer::dumpSwitchFunc_des(raw_indented_ostream &os) {
  raw_indented_ostream::DelimitedScope scope(os);
  os << formatv("switch ({0}): {{\n", inputName);
  {
    raw_indented_ostream::DelimitedScope scope(os);
    for (auto c : cases) {
      os << formatv("case {0}:\n", c.caseType);
      printCodeBlock(os, c.caseBody);
      os << "  break;\n";
    }
    os << "default:\n";
    printCodeBlock(os, "llvm_unreachable(\"NYI\");\n");
    os << "  break;\n";
  }
  os << "}\n";
}

void CppProtoSerializer::genClass() {
  llvm::SmallVector<MethodParameter> ctrParams;
  for (auto param : fields) {
    ctrParams.emplace_back(param.typ, param.name);
    internalClass.addField(param.typ, param.name);
  }

  auto *ctr = internalClass.addConstructor<Method::Inline>(ctrParams);
  for (auto param : fields) {
    ctr->addMemberInitializer(param.name, param.init);
  }

  auto &mainFuncBody =
    internalClass.addMethod(resTy, formatv("{0}{1}", funcName, resTy),
                            MethodParameter(inputTy, inputName))
    ->body();

  dumpSwitchFunc(mainFuncBody.getStream());

  for (auto cas : cases) {
    auto &caseFuncBody =
      internalClass.addMethod(cas.resType,
                              formatv("{0}{1}", funcName, cas.caseTypeName),
                              MethodParameter(cas.caseType, inputName))
      ->body();
    
    printCodeBlock(caseFuncBody.getStream(), cas.translatorBody);
  }
}
