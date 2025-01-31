#include "VespaGen.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>
#include <set>

using namespace vespa;

static std::map<llvm::StringRef, llvm::StringRef> cppTypeToProto = {
    {"ArrayRef<Type>", "repeated MLIRTypeID"},
    {"llvm::ArrayRef<int64_t>", "repeated int64"},
    {"llvm::ArrayRef<bool>", "repeated bool"},
    {"llvm::ArrayRef<mlir::Type>", "repeated MLIRTypeID"},
    {"llvm::ArrayRef<Attribute>", "repeated MLIRAttribute"},
    {"llvm::ArrayRef<NamedAttribute>", "repeated MLIRNamedAttr"},
    {"llvm::ArrayRef<Location>", "repeated MLIRLocation"},

    {"bool", "bool"},
    {"unsigned", "uint32"},
    {"uint64_t", "uint64"},
    {"int", "int32"},
    {"int32_t", "int32"},
    {"int64_t", "int64"},
    {"std::optional<bool>", "bool"},
    {"std::optional<unsigned>", "uint32"},
    {"std::optional<uint64_t>", "uint64"},
    {"std::optional<int64_t>", "int64"},

    {"Type", "MLIRTypeID"},
    {"mlir::Type", "MLIRTypeID"},
    {"cir::StructType", "MLIRTypeID"},
    {"cir::FuncType", "MLIRTypeID"},
    {"cir::BoolType", "MLIRTypeID"},
    {"cir::ComplexType", "MLIRTypeID"},
    {"cir::PointerType", "MLIRTypeID"},
    {"cir::DataMemberType", "MLIRTypeID"},
    {"cir::CIRFPTypeInterface", "MLIRTypeID"},
    {"cir::MethodType", "MLIRTypeID"},

    {"Attribute", "MLIRAttribute"},
    {"mlir::Attribute", "MLIRAttribute"},
    {"mlir::TypedAttr", "MLIRAttribute"},
    {"MemRefLayoutAttrInterface", "MLIRAttribute"},

    {"StringAttr", "MLIRStringAttr"},
    {"mlir::StringAttr", "MLIRStringAttr"},

    {"mlir::ArrayAttr", "MLIRArrayAttr"},
    {"mlir::IntegerAttr", "MLIRIntegerAttr"},
    {"mlir::DictionaryAttr", "MLIRDictionaryAttr"},
    {"mlir::TypeAttr", "MLIRTypeAttr"},
    {"mlir::FlatSymbolRefAttr", "MLIRFlatSymbolRefAttr"},
    {"std::optional<mlir::FlatSymbolRefAttr>", "MLIRFlatSymbolRefAttr"},

    {"NamedAttribute", "MLIRNamedAttr"},

    {"Location", "MLIRLocation"},

    {"cir::GlobalViewAttr", "CIRGlobalViewAttr"},
    {"cir::IntAttr", "CIRIntAttr"},

    {"SignednessSemantics", "MLIRSignednessSemantics"},
    {"CmpOrdering", "CIRCmpOrdering"},
    {"InlineKind", "CIRInlineKind"},
    {"VisibilityKind", "CIRVisibilityKind"},
    {"sob::SignedOverflowBehavior", "CIRSignedOverflowBehavior"},

    {"SourceLanguageAttr", "CIRSourceLanguage"},

    {"APInt", "string"},
    {"llvm::APInt", "string"},
    {"llvm::APFloat", "string"},

    {"llvm::StringRef", "string"},
};

inline static std::set<llvm::StringRef> primitiveSerializable = {
    "bool",
    "unsigned",
    "uint64_t",
    "int",
    "int32_t",
    "int64_t",
    "std::optional<bool>",
    "std::optional<unsigned>",
    "std::optional<uint64_t>",
    "std::optional<int64_t>",
};

inline static std::set<llvm::StringRef> settable = {
    "bool",
    "unsigned",
    "uint64_t",
    "int",
    "int32_t",
    "int64_t",
    "std::optional<bool>",
    "std::optional<unsigned>",
    "std::optional<uint64_t>",
    "std::optional<int64_t>",

    "SignednessSemantics",
    "CmpOrdering",
    "InlineKind",
    "VisibilityKind",
    "sob::SignedOverflowBehavior",
    "SourceLanguageAttr",
};

inline static std::map<llvm::StringRef, llvm::StringRef> cppTypeToSerializer = {
    {"Type", "typeCache.getMLIRTypeID"},
    {"mlir::Type", "typeCache.getMLIRTypeID"},
    {"cir::StructType", "typeCache.getMLIRTypeID"},
    {"cir::FuncType", "typeCache.getMLIRTypeID"},
    {"cir::BoolType", "typeCache.getMLIRTypeID"},
    {"cir::ComplexType", "typeCache.getMLIRTypeID"},
    {"cir::PointerType", "typeCache.getMLIRTypeID"},
    {"cir::DataMemberType", "typeCache.getMLIRTypeID"},
    {"cir::CIRFPTypeInterface", "typeCache.getMLIRTypeID"},
    {"cir::MethodType", "typeCache.getMLIRTypeID"},

    {"Attribute", "attributeSerializer.serializeMLIRAttribute"},
    {"mlir::Attribute", "attributeSerializer.serializeMLIRAttribute"},
    {"mlir::TypedAttr", "attributeSerializer.serializeMLIRAttribute"},
    {"MemRefLayoutAttrInterface", "attributeSerializer.serializeMLIRAttribute"},

    {"StringAttr", "attributeSerializer.serializeMLIRStringAttr"},
    {"mlir::StringAttr", "attributeSerializer.serializeMLIRStringAttr"},

    {"mlir::ArrayAttr", "attributeSerializer.serializeMLIRArrayAttr"},
    {"mlir::IntegerAttr", "attributeSerializer.serializeMLIRIntegerAttr"},
    {"mlir::DictionaryAttr", "attributeSerializer.serializeMLIRDictionaryAttr"},
    {"mlir::TypeAttr", "attributeSerializer.serializeMLIRTypeAttr"},
    {"mlir::FlatSymbolRefAttr", "attributeSerializer.serializeMLIRFlatSymbolRefAttr"},
    {"std::optional<mlir::FlatSymbolRefAttr>", "attributeSerializer.serializeMLIRFlatSymbolRefAttr"},

    {"NamedAttribute", "attributeSerializer.serializeMLIRNamedAttr"},

    {"Location", "attributeSerializer.serializeMLIRLocation"},

    {"cir::GlobalViewAttr", "attributeSerializer.serializeCIRGlobalViewAttr"},
    {"cir::IntAttr", "attributeSerializer.serializeCIRIntAttr"},

    {"SignednessSemantics", "serializeMLIRSignednessSemantics"},
    {"CmpOrdering", "serializeCIRCmpOrdering"},
    {"InlineKind", "serializeCIRInlineKind"},
    {"VisibilityKind", "serializeCIRVisibilityKind"},
    {"sob::SignedOverflowBehavior", "serializeCIRSignedOverflowBehavior"},

    {"SourceLanguageAttr", "serializeCIRSourceLanguage"},

    {"APInt", "serializeAPInt"},
    {"llvm::APInt", "serializeAPInt"},
    {"llvm::APFloat", "serializeAPFloat"},

    {"llvm::StringRef", "serializeStringRef"},
};

static std::map<llvm::StringRef, llvm::StringRef> cppTypeToKotlin = {
    {"ArrayRef<Type>", "ArrayList<MLIRTypeID>"},
    {"llvm::ArrayRef<int64_t>", "ArrayList<Long>"},
    {"llvm::ArrayRef<bool>", "ArrayList<Boolean>"},
    {"llvm::ArrayRef<mlir::Type>", "ArrayList<MLIRTypeID>"},
    {"llvm::ArrayRef<Attribute>", "ArrayList<MLIRAttribute>"},
    {"llvm::ArrayRef<NamedAttribute>", "ArrayList<MLIRNamedAttr>"},
    {"llvm::ArrayRef<Location>", "ArrayList<MLIRLocation>"},

    {"bool", "Boolean"},
    {"unsigned", "Int"},
    {"uint64_t", "Long"},
    {"int", "Int"},
    {"int32_t", "Int"},
    {"int64_t", "Long"},
    {"std::optional<bool>", "Boolean"},
    {"std::optional<unsigned>", "Int"},
    {"std::optional<uint64_t>", "Long"},
    {"std::optional<int64_t>", "Long"},

    {"Type", "MLIRTypeID"},
    {"mlir::Type", "MLIRTypeID"},
    {"cir::StructType", "MLIRTypeID"},
    {"cir::FuncType", "MLIRTypeID"},
    {"cir::BoolType", "MLIRTypeID"},
    {"cir::ComplexType", "MLIRTypeID"},
    {"cir::PointerType", "MLIRTypeID"},
    {"cir::DataMemberType", "MLIRTypeID"},
    {"cir::CIRFPTypeInterface", "MLIRTypeID"},
    {"cir::MethodType", "MLIRTypeID"},

    {"Attribute", "MLIRAttribute"},
    {"mlir::Attribute", "MLIRAttribute"},
    {"mlir::TypedAttr", "MLIRAttribute"},
    {"MemRefLayoutAttrInterface", "MLIRAttribute"},

    {"StringAttr", "MLIRStringAttr"},
    {"mlir::StringAttr", "MLIRStringAttr"},

    {"mlir::ArrayAttr", "MLIRArrayAttr"},
    {"mlir::IntegerAttr", "MLIRIntegerAttr"},
    {"mlir::DictionaryAttr", "MLIRDictionaryAttr"},
    {"mlir::TypeAttr", "MLIRTypeAttr"},
    {"mlir::FlatSymbolRefAttr", "MLIRFlatSymbolRefAttr"},
    {"std::optional<mlir::FlatSymbolRefAttr>", "MLIRFlatSymbolRefAttr"},

    {"NamedAttribute", "MLIRNamedAttr"},

    {"Location", "MLIRLocation"},

    {"cir::GlobalViewAttr", "CIRGlobalViewAttr"},
    {"cir::IntAttr", "CIRIntAttr"},

    {"SignednessSemantics", "MLIRSignednessSemantics"},
    {"CmpOrdering", "CIRCmpOrdering"},
    {"InlineKind", "CIRInlineKind"},
    {"VisibilityKind", "CIRVisibilityKind"},
    {"sob::SignedOverflowBehavior", "CIRSignedOverflowBehavior"},

    {"SourceLanguageAttr", "CIRSourceLanguage"},

    {"APInt", "BigInteger"},
    {"llvm::APInt", "BigInteger"},
    {"llvm::APFloat", "BigDecimal"},

    {"llvm::StringRef", "String"},
};

static std::map<llvm::StringRef, llvm::StringRef> cppTypeToBuilder = {
    {"ArrayRef<Type>", "buildMLIRTypeIDArray"},
    {"llvm::ArrayRef<int64_t>", "buildI64Array"},
    {"llvm::ArrayRef<bool>", "buildBooleanArray"},
    {"llvm::ArrayRef<mlir::Type>", "buildMLIRTypeIDArray"},
    {"llvm::ArrayRef<Attribute>", "buildMLIRAttributeArray"},
    {"llvm::ArrayRef<NamedAttribute>", "buildMLIRNamedAttrArray"},
    {"llvm::ArrayRef<Location>", "buildMLIRLocationArray"},

    {"Type", "buildMLIRTypeID"},
    {"mlir::Type", "buildMLIRTypeID"},
    {"cir::StructType", "buildMLIRTypeID"},
    {"cir::FuncType", "buildMLIRTypeID"},
    {"cir::BoolType", "buildMLIRTypeID"},
    {"cir::ComplexType", "buildMLIRTypeID"},
    {"cir::PointerType", "buildMLIRTypeID"},
    {"cir::DataMemberType", "buildMLIRTypeID"},
    {"cir::CIRFPTypeInterface", "buildMLIRTypeID"},
    {"cir::MethodType", "buildMLIRTypeID"},

    {"Attribute", "buildMLIRAttribute"},
    {"mlir::Attribute", "buildMLIRAttribute"},
    {"mlir::TypedAttr", "buildMLIRAttribute"},
    {"MemRefLayoutAttrInterface", "buildMLIRAttribute"},

    {"StringAttr", "buildMLIRStringAttr"},
    {"mlir::StringAttr", "buildMLIRStringAttr"},

    {"mlir::ArrayAttr", "buildMLIRArrayAttr"},
    {"mlir::IntegerAttr", "buildMLIRIntegerAttr"},
    {"mlir::DictionaryAttr", "buildMLIRDictionaryAttr"},
    {"mlir::TypeAttr", "buildMLIRTypeAttr"},
    {"mlir::FlatSymbolRefAttr", "buildMLIRFlatSymbolRefAttr"},
    {"std::optional<mlir::FlatSymbolRefAttr>", "buildMLIRFlatSymbolRefAttr"},

    {"NamedAttribute", "buildMLIRNamedAttr"},

    {"Location", "buildMLIRLocation"},

    {"cir::GlobalViewAttr", "buildCIRGlobalViewAttr"},
    {"cir::IntAttr", "buildCIRIntAttr"},

    {"SignednessSemantics", "buildMLIRSignednessSemantics"},
    {"CmpOrdering", "buildCIRCmpOrdering"},
    {"InlineKind", "buildCIRInlineKind"},
    {"VisibilityKind", "buildCIRVisibilityKind"},
    {"sob::SignedOverflowBehavior", "buildCIRSignedOverflowBehavior"},

    {"SourceLanguageAttr", "buildCIRSourceLanguage"},

    {"APInt", "BigInteger"},
    {"llvm::APInt", "BigInteger"},
    {"llvm::APFloat", "BigDecimal"},
};

std::string vespa::makeIdentifier(llvm::StringRef str) {
  if (!str.empty() && llvm::isDigit(static_cast<unsigned char>(str.front()))) {
    std::string newStr = std::string("_") + str.str();
    return newStr;
  }
  return str.str();
}

std::string vespa::makeProtoSymbol(llvm::StringRef symbol) {
    return llvm::convertToCamelFromSnakeCase(symbol, true);
}

std::string vespa::makeFullProtoSymbol(llvm::StringRef enumName,
                                    llvm::StringRef symbol) {
    return llvm::formatv("{0}_{1}", enumName, symbol).str();
}

llvm::StringRef removeGlobalScopeQualifier(llvm::StringRef &type) {
  if (type.starts_with("::")) {
    return type.drop_front(2);
  }
  return type;
}

llvm::StringRef removeArray(llvm::StringRef &type) {
  if (type.starts_with("llvm::ArrayRef")) {
    return type.drop_front(15).drop_back(1);
  }
  if (type.starts_with("ArrayRef")) {
    return type.drop_front(9).drop_back(1);
  }
  assert(0);
  return "";
}

llvm::StringRef vespa::getProtoType(mlir::tblgen::AttrOrTypeParameter &p) {
  auto type = p.getCppType();
  return cppTypeToProto.at(removeGlobalScopeQualifier(type));
}

llvm::StringRef vespa::getKotlinType(mlir::tblgen::AttrOrTypeParameter &p) {
  auto type = p.getCppType();
  return cppTypeToKotlin.at(removeGlobalScopeQualifier(type));
}

void vespa::serializeParameter(mlir::tblgen::AttrOrTypeParameter &p,
                        llvm::StringRef varName,
                        llvm::raw_ostream &os) {
  auto name = p.getName();
  auto snake = llvm::convertToSnakeFromCamelCase(name);
  auto getter = llvm::convertToCamelFromSnakeCase("get_" + snake);
  auto type = p.getCppType();
  type = removeGlobalScopeQualifier(type);

  std::string field = llvm::formatv("{0}.{1}()", varName, getter);

  if (type == "SourceLanguageAttr") {
    field = llvm::formatv("{0}.getValue()", field);
  }

  if (type.starts_with("std::optional")) {
    field = llvm::formatv("*{0}", field);
  }

  if (type.starts_with("llvm::ArrayRef") || type.starts_with("ArrayRef")) {
    type = removeArray(type);
    std::string i = "i";
    if (cppTypeToSerializer.count(type)) {
      i = llvm::formatv("{0}({1})", cppTypeToSerializer.at(type), i);
    } else {
      assert(primitiveSerializable.count(type));
    }

    // hotfix
    if (varName == "attr" && llvm::StringRef(i).starts_with("attributeSerializer")) {
      i = llvm::StringRef(i).drop_front(20).str();
    }

    os << llvm::formatv("  for (auto i : {0}) {{\n", field);
    os << llvm::formatv("    serialized.mutable_{0}()->Add({1});\n", snake, i);
    os << llvm::formatv("  }\n");
    return;
  }

  if (cppTypeToSerializer.count(type)) {
    field = llvm::formatv("{0}({1})", cppTypeToSerializer.at(type), field);
  } else {
    assert(primitiveSerializable.count(type));
  }

  // hotfix
  if (varName == "attr" && llvm::StringRef(field).starts_with("attributeSerializer")) {
    field = llvm::StringRef(field).drop_front(20).str();
  }

  std::string padding = "";
  if (p.isOptional()) {
    padding = "  ";
    os << llvm::formatv("  if ({0}.{1}()) {{\n", varName, getter);
  }

  if (settable.count(type)) {
    os << llvm::formatv("{0}  serialized.set_{1}({2});\n", padding, snake, field);
  } else {
    os << llvm::formatv("{0}  *serialized.mutable_{1}() = {2};\n", padding, snake, field);
  }

  if (p.isOptional()) {
    os << "  }\n";
  }
}

void vespa::buildParameter(mlir::tblgen::AttrOrTypeParameter &p,
                    llvm::StringRef varName,
                    llvm::raw_ostream &os,
                    size_t padding) {
  auto camel = llvm::convertToCamelFromSnakeCase(p.getName(), false);
  auto upperCamel = llvm::convertToCamelFromSnakeCase(p.getName(), true);
  auto assign = llvm::formatv("{0}.{1}", varName, camel).str();

  auto type = p.getCppType();
  type = removeGlobalScopeQualifier(type);
  if (cppTypeToBuilder.count(type)) {
    if (type.starts_with("ArrayRef") || type.starts_with("llvm::ArrayRef")) {
      assign = llvm::formatv("{0}List", assign);
    }
    assign = llvm::formatv("{0}({1})", cppTypeToBuilder.at(type), assign);
  }

  std::string pad(padding, ' ');
  if (p.isOptional()) {
    os << llvm::formatv("{0}if ({1}.has{2}()) {3} else null,\n", pad, varName, upperCamel, assign);
  } else {
    os << llvm::formatv("{0}{1},\n", pad, assign);
  }
}
