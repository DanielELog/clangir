//===- ProtoOpSerializerGen.cpp - Proto op serializer generator -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ProtoOpDefinitionsGen uses the description of operations to generate Proto
// definitions for ops.
//
//===----------------------------------------------------------------------===//

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

#include <map>
#include <set>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using llvm::formatv;
using llvm::RecordKeeper;

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

const char *const serializerFileHeader = R"(
#include "cir-tac/EnumsSerializer.h"
#include "cir-tac/Serializer.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace protocir;
)";

const char *const serializerDefStart = R"(
protocir::CIROp Serializer::serializeOperation(mlir::Operation &inst,
                                               protocir::CIRModuleID pModuleID,
                                               TypeCache &typeCache,
                                               OperationCache &opCache,
                                               BlockCache &blockCache) {
  protocir::CIROp pInst;

  auto resultTypes = inst.getResultTypes();
  for (const auto &resultType : resultTypes) {
    auto resultTypeID = internType(typeCache, resultType);
    pInst.add_result_types()->set_id(resultTypeID);
  }

  auto instID = internOperation(opCache, &inst);
  llvm::TypeSwitch<mlir::Operation *>(&inst)
)";

const char *const serializerDefEnd = R"(
  return pInst;
}
)";

const char *const serializerCaseStart = R"(
      .Case<cir::{0}>([instID, &pInst, pModuleID, &typeCache, &blockCache, &opCache](cir::{0} op) {{
        protocir::CIR{0} p{0};
        pInst.mutable_base()->set_id(instID);
)";

const char *const serializerCaseDefineOptionalOperation = R"(
        auto {0}Raw = op.{1}();
        if ({0}Raw) {{
          auto {0}Value = Serializer::serializeValue(
              {0}Raw, pModuleID, typeCache, opCache, blockCache);
          *p{2}.mutable_{3}() = {0}Value;
        }
)";

const char *const serializerCaseDefineVariadicOperation = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          auto e{0}Value = Serializer::serializeValue(
              e{0}, pModuleID, typeCache, opCache, blockCache);
          e{0}Proto->CopyFrom(e{0}Value);
        }
)";

const char *const serializerCaseDefineVariadicOfVariadicOperation = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          for (auto ee{0} : e{0}) {{
            auto ee{0}Proto = e{0}Proto->add_range();
            auto ee{0}Value = Serializer::serializeValue(
                ee{0}, pModuleID, typeCache, opCache, blockCache);
            ee{0}Proto->CopyFrom(ee{0}Value);
          }
        }
)";

const char *const serializerCaseDefineOperation = R"(
        auto {0} = op.{1}();
        auto {0}Value = Serializer::serializeValue(
            {0}, pModuleID, typeCache, opCache, blockCache);
        *p{2}.mutable_{3}() = {0}Value;
)";

const char *const serializerCaseDefinePrimitive = R"(
        auto {0} = op.{1}();
        p{2}.set_{3}({0});
)";

const char *const serializerCaseDefineType = R"(
        auto {0} = op.{1}();
        auto {0}ID = internType(typeCache, {0});

        protocir::CIRTypeID p{0}ID;
        p{0}ID.set_id({0}ID);

        *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefineOptional = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          *p{2}.mutable_{3}() = {0};
        }
)";

const char *const serializerCaseDefineOptionalPrimitive = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          p{2}.set_{3}({0});
        }
)";

const char *const serializerCaseDefineOptionalCtorDtor = R"(
        auto {0}Optional = op.{1}();
        p{2}.set_{3}({0}Optional.has_value());
)";

const char *const serializerCaseDefine = R"(
        auto {0} = op.{1}();
        *p{2}.mutable_{3}() = {0};
)";

const char *const serializerCaseDefineAPInt = R"(
        llvm::SmallVector<char> {0}Str;
        auto {0} = op.{1}();
        {0}.toString({0}Str, 10, false); 
        llvm::StringRef {0}StrRef({0}Str.data(), {0}Str.size());
        *p{2}.mutable_{3}() = {0}StrRef;

)";

const char *const serializerCaseDefineOptionalAPFloat = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          llvm::SmallVector<char> {0}Str;
          {0}.toString({0}Str); 
          llvm::StringRef {0}StrRef({0}Str.data(), {0}Str.size());
          *p{2}.mutable_{3}() = {0}StrRef;
        }
)";

const char *const serializerCaseDefineTypedAttr = R"(
        std::string {0}Str;
        llvm::raw_string_ostream {0}RawStream({0}Str);
        op.{1}().print({0}RawStream);
        *p{2}.mutable_{3}() = {0}Str;
)";

const char *const serializerCaseDefineVariadicPrimitive = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          p{2}.add_{3}(e{0});
        }
)";

const char *const serializerCaseDefineEnum = R"(
        auto {0} = op.{1}();
        auto p{0} = EnumSerializer::serialize{4}({0});
        p{2}.set_{3}(p{0});
)";

const char *const serializerCaseDefineEnumAttr = R"(
        auto {0} = op.{1}().getValue();
        auto p{0} = EnumSerializer::serialize{4}({0});
        p{2}.set_{3}(p{0});
)";

const char *const serializerCaseDefineOptionalEnum = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();
          auto p{0} = EnumSerializer::serialize{4}({0});
          p{2}.set_{3}(p{0});
        }
)";

const char *const serializerCaseDefineOptionalAttribute = R"(
        auto {0}Optional = op.{1}();
        if ({0}Optional) {{
          auto {0} = {0}Optional.value();

          std::string {0}Str;
          llvm::raw_string_ostream {0}RawStream({0}Str);
          {0}.print({0}RawStream);
          *p{2}.mutable_{3}() = {0}Str;
        }
)";

const char *const serializerCaseDefineSuccessor = R"(
        auto {0} = op.{1}();
        auto {0}ID = internBlock(blockCache, {0});
        protocir::CIRBlockID p{0}ID;
        p{0}ID.set_id({0}ID);

        *p{2}.mutable_{3}() = p{0}ID;
)";

const char *const serializerCaseDefineVariadicSuccessor = R"(
        auto {0} = op.{1}();
        for (auto e{0} : {0}) {{
          auto e{0}Proto = p{2}.add_{3}();
          auto e{0}ID = internBlock(blockCache, e{0});
          protocir::CIRBlockID pe{0}ID;
          pe{0}ID.set_id(e{0}ID);
        }
)";

const char *const serializerCaseEnd = R"(
        pInst.mutable_{0}()->CopyFrom(p{1});
      })
)";

const char *const serializerDefaultCase = R"(
      .Default([](mlir::Operation *op) {
        op->dump();
        llvm_unreachable("NIY");
      });
)";

const std::map<StringRef, StringRef> cppAttrTypeToProto = {
    {"uint64_t", "uint64"},
    {"uint32_t", "uint32"},
    {"::llvm::StringRef", "string"},
    {"::llvm::APInt", "string"},
    {"::llvm::APFloat", "string"},
    {"::cir::GlobalDtorAttr", "bool"},
    {"::cir::GlobalCtorAttr", "bool"},
    {"::llvm::ArrayRef<int32_t>", "repeated uint32"},
    {"::mlir::Attribute", "string"},
    {"::mlir::TypedAttr", "string"},
    {"::cir::VisibilityAttr", "CIRVisibilityKind"},
    {"::cir::FuncType", "CIROpID"},
    {"::mlir::Type", "CIRTypeID"},
    {"::cir::PointerType", "CIROpID"},
    {"::cir::IntType", "CIROpID"},
    {"::cir::MethodType", "CIROpID"},
    {"::cir::DataMemberType", "CIROpID"},
    {"::cir::ComplexType", "CIROpID"},
    {"::cir::VectorType", "CIROpID"},
    {"::cir::BoolType", "CIROpID"}};

const std::map<StringRef, StringRef> cppOperandTypeToProto = {
    {"uint64_t", "uint64"},
    {"uint32_t", "uint32"},
    {"::llvm::StringRef", "string"},
    {"::llvm::APInt", "string"},
    {"::llvm::APFloat", "string"},
    {"::llvm::ArrayRef<int32_t>", "repeated uint32"},
    {"::mlir::TypedAttr", "string"},
    {"::cir::VisibilityAttr", "CIRVisibilityKind"},
    {"::cir::FuncType", "CIROpID"},
    {"::mlir::Type", "CIROpID"},
    {"::cir::PointerType", "CIROpID"},
    {"::cir::IntType", "CIROpID"},
    {"::cir::MethodType", "CIROpID"},
    {"::cir::DataMemberType", "CIROpID"},
    {"::cir::ComplexType", "CIROpID"},
    {"::cir::VectorType", "CIROpID"},
    {"::cir::BoolType", "CIROpID"}};

const std::set<StringRef> typesBlackList = {
    "::std::optional< ::mlir::ArrayAttr >",
    "::std::optional<::cir::DynamicCastInfoAttr>",
    "::std::optional<::cir::ASTVarDeclInterface>",
    "::mlir::ArrayAttr",
    "::cir::CmpThreeWayInfoAttr",
    "::cir::BitfieldInfoAttr",
    "::std::optional<::cir::AddressSpaceAttr>",
    "::std::optional<::cir::ASTCallExprInterface>",
    "::cir::ExtraFuncAttributesAttr"};

static void emitOptionalAttributeSerializer(
    Operator &op, const llvm::StringRef &attrName,
    const llvm::StringRef &attrNameCpp, const llvm::StringRef &attrNameProto,
    const llvm::StringRef &attrType, const llvm::StringRef &attrTypeProto,
    raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  if (attrType == "bool") {
    os << formatv(serializerCaseDefinePrimitive, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::APFloat") {
    os << formatv(serializerCaseDefineOptionalAPFloat, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "uint32_t" || attrType == "uint64_t") {
    os << formatv(serializerCaseDefineOptionalPrimitive, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else if (attrType == "::cir::GlobalCtorAttr" ||
             attrType == "::cir::GlobalDtorAttr") {
    os << formatv(serializerCaseDefineOptionalCtorDtor, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::Attribute") {
    os << formatv(serializerCaseDefineOptionalAttribute, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else {
    os << formatv(serializerCaseDefineOptional, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  }
}

static void emitEnumAttributeSerializer(Operator &op,
                                        const llvm::StringRef &attrName,
                                        const llvm::StringRef &enumName,
                                        const llvm::StringRef &attrNameCpp,
                                        const llvm::StringRef &attrNameProto,
                                        raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  os << formatv(serializerCaseDefineEnum, attrNameCpp, getterName,
                op.getCppClassName(), attrNameProto, enumName);
}

static void emitAttributeSerializer(Operator &op,
                                    const llvm::StringRef &attrName,
                                    const llvm::StringRef &attrNameCpp,
                                    const llvm::StringRef &attrNameProto,
                                    const llvm::StringRef &attrType,
                                    const llvm::StringRef &attrTypeProto,
                                    raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  if (attrType == "bool" || attrType == "uint32_t" || attrType == "uint64_t") {
    os << formatv(serializerCaseDefinePrimitive, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::TypedAttr") {
    os << formatv(serializerCaseDefineTypedAttr, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::APInt") {
    os << formatv(serializerCaseDefineAPInt, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::llvm::ArrayRef<int32_t>") {
    os << formatv(serializerCaseDefineVariadicPrimitive, attrNameCpp,
                  getterName, op.getCppClassName(), attrNameProto);
  } else if (attrType == "::mlir::Type" || attrType == "::cir::FuncType") {
    os << formatv(serializerCaseDefineType, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  } else if (attrType == "::cir::VisibilityAttr") {
    std::string getterName = op.getGetterName(attrName);
    os << formatv(serializerCaseDefineEnumAttr, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto, "VisibilityKind");
  } else {
    os << formatv(serializerCaseDefine, attrNameCpp, getterName,
                  op.getCppClassName(), attrNameProto);
  }
}

static void emitOptionalEnumAttributeSerializer(
    Operator &op, const llvm::StringRef &attrName,
    const llvm::StringRef &enumName, const llvm::StringRef &attrNameCpp,
    const llvm::StringRef &attrNameProto, raw_ostream &os) {
  std::string getterName = op.getGetterName(attrName);
  os << formatv(serializerCaseDefineOptionalEnum, attrNameCpp, getterName,
                op.getCppClassName(), attrNameProto, enumName);
}

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (!operand.name.empty())
    return std::string(operand.name);
  return std::string(formatv("{0}_{1}", generatedArgName, index));
}

// Replaces all occurrences of `match` in `str` with `substitute`.
static std::string replaceAllSubstrs(std::string str, const std::string &match,
                                     const std::string &substitute) {
  std::string::size_type scanLoc = 0, matchLoc = std::string::npos;
  while ((matchLoc = str.find(match, scanLoc)) != std::string::npos) {
    str = str.replace(matchLoc, match.size(), substitute);
    scanLoc = matchLoc + substitute.size();
  }
  return str;
}

// Returns true if we can use unwrapped value for the given `attr` in builders.
static bool canUseUnwrappedRawValue(const tblgen::Attribute &attr) {
  return attr.getReturnType() != attr.getStorageType() &&
         // We need to wrap the raw value into an attribute in the builder impl
         // so we need to make sure that the attribute specifies how to do that.
         !attr.getConstBuilderTemplate().empty();
}

/// Build an attribute from a parameter value using the constant builder.
static std::string constBuildAttrFromParam(const tblgen::Attribute &attr,
                                           FmtContext &fctx,
                                           StringRef paramName) {
  std::string builderTemplate = attr.getConstBuilderTemplate().str();

  // For StringAttr, its constant builder call will wrap the input in
  // quotes, which is correct for normal string literals, but incorrect
  // here given we use function arguments. So we need to strip the
  // wrapping quotes.
  if (StringRef(builderTemplate).contains("\"$0\""))
    builderTemplate = replaceAllSubstrs(builderTemplate, "\"$0\"", "$0");

  return tgfmt(builderTemplate, &fctx, paramName).str();
}

static void genCodeForAddingArgAndRegionForBuilder(Operator &op,
    mlir::tblgen::MethodBody &body, llvm::StringSet<> &inferredAttributes,
    bool isRawValueAttr) {
  // Push all operands to the result.
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    std::string argName = getArgumentName(op, i);
    const NamedTypeConstraint &operand = op.getOperand(i);
    if (operand.constraint.isVariadicOfVariadic()) {
      body << "  for (::mlir::ValueRange range : " << argName << ")\n   "
           << builderOpState << ".addOperands(range);\n";

      // Add the segment attribute.
      body << "  {\n"
           << "    ::llvm::SmallVector<int32_t> rangeSegments;\n"
           << "    for (::mlir::ValueRange range : " << argName << ")\n"
           << "      rangeSegments.push_back(range.size());\n"
           << "    auto rangeAttr = " << odsBuilder
           << ".getDenseI32ArrayAttr(rangeSegments);\n";
      if (op.getDialect().usePropertiesForAttributes()) {
        body << "    " << builderOpState << ".getOrAddProperties<Properties>()."
             << operand.constraint.getVariadicOfVariadicSegmentSizeAttr()
             << " = rangeAttr;";
      } else {
        body << "    " << builderOpState << ".addAttribute("
             << op.getGetterName(
                    operand.constraint.getVariadicOfVariadicSegmentSizeAttr())
             << "AttrName(" << builderOpState << ".name), rangeAttr);";
      }
      body << "  }\n";
      continue;
    }

    if (operand.isOptional())
      body << "  if (" << argName << ")\n  ";
    body << "  " << builderOpState << ".addOperands(" << argName << ");\n";
  }

  // If the operation has the operand segment size attribute, add it here.
  auto emitSegment = [&]() {
    llvm::interleaveComma(llvm::seq<int>(0, op.getNumOperands()), body, [&](int i) {
      const NamedTypeConstraint &operand = op.getOperand(i);
      if (!operand.isVariableLength()) {
        body << "1";
        return;
      }

      std::string operandName = getArgumentName(op, i);
      if (operand.isOptional()) {
        body << "(" << operandName << " ? 1 : 0)";
      } else if (operand.isVariadicOfVariadic()) {
        body << llvm::formatv(
            "static_cast<int32_t>(std::accumulate({0}.begin(), {0}.end(), 0, "
            "[](int32_t curSum, ::mlir::ValueRange range) {{ return curSum + "
            "static_cast<int32_t>(range.size()); }))",
            operandName);
      } else {
        body << "static_cast<int32_t>(" << getArgumentName(op, i) << ".size())";
      }
    });
  };
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::string sizes = op.getGetterName(operandSegmentAttrName);
    if (op.getDialect().usePropertiesForAttributes()) {
      body << "  ::llvm::copy(::llvm::ArrayRef<int32_t>({";
      emitSegment();
      body << "}), " << builderOpState
           << ".getOrAddProperties<Properties>()."
              "operandSegmentSizes.begin());\n";
    } else {
      body << "  " << builderOpState << ".addAttribute(" << sizes << "AttrName("
           << builderOpState << ".name), "
           << "odsBuilder.getDenseI32ArrayAttr({";
      emitSegment();
      body << "}));\n";
    }
  }

  // Push all properties to the result.
  for (const auto &namedProp : op.getProperties()) {
    // Use the setter from the Properties struct since the conversion from the
    // interface type (used in the builder argument) to the storage type (used
    // in the state) is not necessarily trivial.
    std::string setterName = op.getSetterName(namedProp.name);
    body << formatv("  {0}.getOrAddProperties<Properties>().{1}({2});\n",
                    builderOpState, setterName, namedProp.name);
  }
  // Push all attributes to the result.
  for (const auto &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr() || inferredAttributes.contains(namedAttr.name))
      continue;

    // TODO: The wrapping of optional is different for default or not, so don't
    // unwrap for default ones that would fail below.
    bool emitNotNullCheck =
        (attr.isOptional() && !attr.hasDefaultValue()) ||
        (attr.hasDefaultValue() && !isRawValueAttr) ||
        // TODO: UnitAttr is optional, not wrapped, but needs to be guarded as
        // the constant materialization is only for true case.
        (isRawValueAttr && attr.getAttrDefName() == "UnitAttr");
    if (emitNotNullCheck)
      body.indent() << formatv("if ({0}) ", namedAttr.name) << "{\n";

    if (isRawValueAttr && canUseUnwrappedRawValue(attr)) {
      // If this is a raw value, then we need to wrap it in an Attribute
      // instance.
      FmtContext fctx;
      fctx.withBuilder("odsBuilder");
      if (op.getDialect().usePropertiesForAttributes()) {
        body << formatv("  {0}.getOrAddProperties<Properties>().{1} = {2};\n",
                        builderOpState, namedAttr.name,
                        constBuildAttrFromParam(attr, fctx, namedAttr.name));
      } else {
        body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                        builderOpState, op.getGetterName(namedAttr.name),
                        constBuildAttrFromParam(attr, fctx, namedAttr.name));
      }
    } else {
      if (op.getDialect().usePropertiesForAttributes()) {
        body << formatv("  {0}.getOrAddProperties<Properties>().{1} = {1};\n",
                        builderOpState, namedAttr.name);
      } else {
        body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                        builderOpState, op.getGetterName(namedAttr.name),
                        namedAttr.name);
      }
    }
    if (emitNotNullCheck)
      body.unindent() << "  }\n";
  }

  // Create the correct number of regions.
  for (const NamedRegion &region : op.getRegions()) {
    if (region.isVariadic())
      body << formatv("  for (unsigned i = 0; i < {0}Count; ++i)\n  ",
                      region.name);

    body << "  (void)" << builderOpState << ".addRegion();\n";
  }

  // Push all successors to the result.
  for (const NamedSuccessor &namedSuccessor : op.getSuccessors()) {
    body << formatv("  {0}.addSuccessors({1});\n", builderOpState,
                    namedSuccessor.name);
  }
}

// The kind of parameter to generate for result types in builders.
enum class TypeParamKind {
None,       // No result type in parameter list.
Separate,   // A separate parameter for each result type.
Collective, // An ArrayRef<Type> for all result types.
};

// The kind of parameter to generate for attributes in builders.
enum class AttrParamKind {
WrappedAttr,    // A wrapped MLIR Attribute instance.
UnwrappedValue, // A raw value without MLIR Attribute wrapper.
};

static bool canGenerateUnwrappedBuilder(const Operator &op) {
  // If this op does not have native attributes at all, return directly to avoid
  // redefining builders.
  if (op.getNumNativeAttributes() == 0)
    return false;

  bool canGenerate = false;
  // We are generating builders that take raw values for attributes. We need to
  // make sure the native attributes have a meaningful "unwrapped" value type
  // different from the wrapped mlir::Attribute type to avoid redefining
  // builders. This checks for the op has at least one such native attribute.
  for (int i = 0, e = op.getNumNativeAttributes(); i < e; ++i) {
    const NamedAttribute &namedAttr = op.getAttribute(i);
    if (canUseUnwrappedRawValue(namedAttr.attr)) {
      canGenerate = true;
      break;
    }
  }
  return canGenerate;
}

static void buildParamList(Operator &op, SmallVectorImpl<MethodParameter> &paramList,
                               llvm::StringSet<> &inferredAttributes,
                               SmallVectorImpl<std::string> &resultTypeNames,
                               TypeParamKind typeParamKind,
                               AttrParamKind attrParamKind) {
  resultTypeNames.clear();
  auto numResults = op.getNumResults();
  resultTypeNames.reserve(numResults);

  paramList.emplace_back("::mlir::OpBuilder &", odsBuilder);
  paramList.emplace_back("::mlir::OperationState &", builderOpState);

  switch (typeParamKind) {
  case TypeParamKind::None:
    break;
  case TypeParamKind::Separate: {
    // Add parameters for all return types
    for (int i = 0; i < numResults; ++i) {
      const auto &result = op.getResult(i);
      std::string resultName = std::string(result.name);
      if (resultName.empty())
        resultName = std::string(formatv("resultType{0}", i));

      StringRef type =
          result.isVariadic() ? "::mlir::TypeRange" : "::mlir::Type";

      paramList.emplace_back(type, resultName, result.isOptional());
      resultTypeNames.emplace_back(std::move(resultName));
    }
  } break;
  case TypeParamKind::Collective: {
    paramList.emplace_back("::mlir::TypeRange", "resultTypes");
    resultTypeNames.push_back("resultTypes");
  } break;
  }

  // Add parameters for all arguments (operands and attributes).
  // Track "attr-like" (property and attribute) optional values separate from
  // attributes themselves so that the disambiguation code can look at the first
  // attribute specifically when determining where to trim the optional-value
  // list to avoid ambiguity while preserving the ability of all-property ops to
  // use default parameters.
  int defaultValuedAttrLikeStartIndex = op.getNumArgs();
  int defaultValuedAttrStartIndex = op.getNumArgs();
  // Successors and variadic regions go at the end of the parameter list, so no
  // default arguments are possible.
  bool hasTrailingParams = op.getNumSuccessors() || op.getNumVariadicRegions();
  if (!hasTrailingParams) {
    // Calculate the start index from which we can attach default values in the
    // builder declaration.
    for (int i = op.getNumArgs() - 1; i >= 0; --i) {
      auto *namedAttr =
          llvm::dyn_cast_if_present<tblgen::NamedAttribute *>(op.getArg(i));
      auto *namedProperty =
          llvm::dyn_cast_if_present<tblgen::NamedProperty *>(op.getArg(i));
      if (namedProperty) {
        Property prop = namedProperty->prop;
        if (!prop.hasDefaultValue())
          break;
        defaultValuedAttrLikeStartIndex = i;
        continue;
      }
      if (!namedAttr)
        break;

      Attribute attr = namedAttr->attr;
      // TODO: Currently we can't differentiate between optional meaning do not
      // verify/not always error if missing or optional meaning need not be
      // specified in builder. Expand isOptional once we can differentiate.
      if (!attr.hasDefaultValue() && !attr.isDerivedAttr())
        break;

      // Creating an APInt requires us to provide bitwidth, value, and
      // signedness, which is complicated compared to others. Similarly
      // for APFloat.
      // TODO: Adjust the 'returnType' field of such attributes
      // to support them.
      StringRef retType = namedAttr->attr.getReturnType();
      if (retType == "::llvm::APInt" || retType == "::llvm::APFloat")
        break;

      defaultValuedAttrLikeStartIndex = i;
      defaultValuedAttrStartIndex = i;
    }
  }
  // Avoid generating build methods that are ambiguous due to default values by
  // requiring at least one attribute.
  if (defaultValuedAttrStartIndex < op.getNumArgs()) {
    // TODO: This should have been possible as a cast<NamedAttribute> but
    // required template instantiations is not yet defined for the tblgen helper
    // classes.
    auto *namedAttr =
        cast<NamedAttribute *>(op.getArg(defaultValuedAttrStartIndex));
    Attribute attr = namedAttr->attr;
    if ((attrParamKind == AttrParamKind::WrappedAttr &&
         canUseUnwrappedRawValue(attr)) ||
        (attrParamKind == AttrParamKind::UnwrappedValue &&
         !canUseUnwrappedRawValue(attr))) {
      ++defaultValuedAttrStartIndex;
      defaultValuedAttrLikeStartIndex = defaultValuedAttrStartIndex;
    }
  }

  /// Collect any inferred attributes.
  for (const NamedTypeConstraint &operand : op.getOperands()) {
    if (operand.isVariadicOfVariadic()) {
      inferredAttributes.insert(
          operand.constraint.getVariadicOfVariadicSegmentSizeAttr());
    }
  }

  for (int i = 0, e = op.getNumArgs(), numOperands = 0; i < e; ++i) {
    Argument arg = op.getArg(i);
    if (const auto *operand =
            llvm::dyn_cast_if_present<NamedTypeConstraint *>(arg)) {
      StringRef type;
      if (operand->isVariadicOfVariadic())
        type = "::llvm::ArrayRef<::mlir::ValueRange>";
      else if (operand->isVariadic())
        type = "::mlir::ValueRange";
      else
        type = "::mlir::Value";

      paramList.emplace_back(type, getArgumentName(op, numOperands++),
                             operand->isOptional());
      continue;
    }
    if (auto *propArg = llvm::dyn_cast_if_present<NamedProperty *>(arg)) {
      const Property &prop = propArg->prop;
      StringRef type = prop.getInterfaceType();
      std::string defaultValue;
      if (prop.hasDefaultValue() && i >= defaultValuedAttrLikeStartIndex) {
        defaultValue = prop.getDefaultValue();
      }
      bool isOptional = prop.hasDefaultValue();
      paramList.emplace_back(type, propArg->name, StringRef(defaultValue),
                             isOptional);
      continue;
    }
    const NamedAttribute &namedAttr = *arg.get<NamedAttribute *>();
    const Attribute &attr = namedAttr.attr;

    // Inferred attributes don't need to be added to the param list.
    if (inferredAttributes.contains(namedAttr.name))
      continue;

    StringRef type;
    switch (attrParamKind) {
    case AttrParamKind::WrappedAttr:
      type = attr.getStorageType();
      break;
    case AttrParamKind::UnwrappedValue:
      if (canUseUnwrappedRawValue(attr))
        type = attr.getReturnType();
      else
        type = attr.getStorageType();
      break;
    }

    // Attach default value if requested and possible.
    std::string defaultValue;
    if (i >= defaultValuedAttrStartIndex) {
      if (attrParamKind == AttrParamKind::UnwrappedValue &&
          canUseUnwrappedRawValue(attr))
        defaultValue += attr.getDefaultValue();
      else
        defaultValue += "nullptr";
    }
    paramList.emplace_back(type, namedAttr.name, StringRef(defaultValue),
                           attr.isOptional());
  }

  /// Insert parameters for each successor.
  for (const NamedSuccessor &succ : op.getSuccessors()) {
    StringRef type =
        succ.isVariadic() ? "::mlir::BlockRange" : "::mlir::Block *";
    paramList.emplace_back(type, succ.name);
  }

  /// Insert parameters for variadic regions.
  for (const NamedRegion &region : op.getRegions())
    if (region.isVariadic())
      paramList.emplace_back("unsigned",
                             llvm::formatv("{0}Count", region.name).str());
}

static bool canInferType(const Operator &op) {
  return op.getTrait("::mlir::InferTypeOpInterface::Trait");
}

static void genSeparateArgParamBuilder(Operator &op, raw_ostream &os) {
  SmallVector<AttrParamKind, 2> attrBuilderType;
  attrBuilderType.push_back(AttrParamKind::WrappedAttr);
  if (canGenerateUnwrappedBuilder(op))
    attrBuilderType.push_back(AttrParamKind::UnwrappedValue);

  // Emit with separate builders with or without unwrapped attributes and/or
  // inferring result type.
  auto emit = [&](AttrParamKind attrType, TypeParamKind paramKind,
                  bool inferType) {
    SmallVector<MethodParameter> paramList;
    SmallVector<std::string, 4> resultNames;
    llvm::StringSet<> inferredAttributes;
    buildParamList(op, paramList, inferredAttributes, resultNames, paramKind,
                   attrType);

    auto m = Method("void", "build", Method::Static, paramList);
    auto &body = m.body();
    genCodeForAddingArgAndRegionForBuilder(op, body, inferredAttributes,
                                           /*isRawValueAttr=*/attrType ==
                                               AttrParamKind::UnwrappedValue);

    // Push all result types to the operation state

    if (inferType) {
      // Generate builder that infers type too.
      // TODO: Subsume this with general checking if type can be
      // inferred automatically.
      body << formatv(R"(
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
                      {1}.location, {1}.operands,
                      {1}.attributes.getDictionary({1}.getContext()),
                      {1}.getRawProperties(),
                      {1}.regions, inferredReturnTypes)))
          {1}.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError({1});
        )",
                      op.getCppClassName(), builderOpState);

      return;
    }

    switch (paramKind) {
    case TypeParamKind::None:
      break;
    case TypeParamKind::Separate:
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        if (op.getResult(i).isOptional())
          body << "  if (" << resultNames[i] << ")\n  ";
        body << "  " << builderOpState << ".addTypes(" << resultNames[i]
             << ");\n";
      }

      // Automatically create the 'resultSegmentSizes' attribute using
      // the length of the type ranges.
      if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
        if (op.getDialect().usePropertiesForAttributes()) {
          body << "  ::llvm::copy(::llvm::ArrayRef<int32_t>({";
        } else {
          std::string getterName = op.getGetterName(resultSegmentAttrName);
          body << " " << builderOpState << ".addAttribute(" << getterName
               << "AttrName(" << builderOpState << ".name), "
               << "odsBuilder.getDenseI32ArrayAttr({";
        }
        interleaveComma(
            llvm::seq<int>(0, op.getNumResults()), body, [&](int i) {
              const NamedTypeConstraint &result = op.getResult(i);
              if (!result.isVariableLength()) {
                body << "1";
              } else if (result.isOptional()) {
                body << "(" << resultNames[i] << " ? 1 : 0)";
              } else {
                // VariadicOfVariadic of results are currently unsupported in
                // MLIR, hence it can only be a simple variadic.
                // TODO: Add implementation for VariadicOfVariadic results here
                //       once supported.
                assert(result.isVariadic());
                body << "static_cast<int32_t>(" << resultNames[i] << ".size())";
              }
            });
        if (op.getDialect().usePropertiesForAttributes()) {
          body << "}), " << builderOpState
               << ".getOrAddProperties<Properties>()."
                  "resultSegmentSizes.begin());\n";
        } else {
          body << "}));\n";
        }
      }

      break;
    case TypeParamKind::Collective: {
      int numResults = op.getNumResults();
      int numVariadicResults = op.getNumVariableLengthResults();
      int numNonVariadicResults = numResults - numVariadicResults;
      bool hasVariadicResult = numVariadicResults != 0;

      // Avoid emitting "resultTypes.size() >= 0u" which is always true.
      if (!hasVariadicResult || numNonVariadicResults != 0)
        body << "  "
             << "assert(resultTypes.size() "
             << (hasVariadicResult ? ">=" : "==") << " "
             << numNonVariadicResults
             << "u && \"mismatched number of results\");\n";
      body << "  " << builderOpState << ".addTypes(resultTypes);\n";
    }
      break;
    default:
      llvm_unreachable("unhandled TypeParamKind");
    }
    auto out = mlir::raw_indented_ostream(os);
    m.writeDefTo(out, "");
  };

  // Some of the build methods generated here may be ambiguous, but TableGen's
  // ambiguous function detection will elide those ones.
  for (auto attrType : attrBuilderType) {
    emit(attrType, TypeParamKind::Separate, /*inferType=*/false);
    if (canInferType(op))
      emit(attrType, TypeParamKind::None, /*inferType=*/true);
    emit(attrType, TypeParamKind::Collective, /*inferType=*/false);
  }
}

static bool emitOpProtoSerializer(const RecordKeeper &records,
                                  raw_ostream &os) {
  os << "/* Autogenerated by mlir-tblgen; don't manually edit. */\n";
  std::vector<const Record *> defs = getRequestedOpDefinitions(records);
  for (auto *def : defs) {
    Operator op(*def);
    if (op.skipDefaultBuilders()) {
        genSeparateArgParamBuilder(op, os);
    }
  }
  return false;
}

static mlir::GenRegistration
    genOpSerializerProto("gen-op-deser-proto",
                         "Generate serializer to op Proto definitions",
                         &emitOpProtoSerializer);
