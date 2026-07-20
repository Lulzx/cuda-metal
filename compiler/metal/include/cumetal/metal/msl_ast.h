#pragma once

#include "cumetal/ir/ir.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cumetal::metal {

enum class MslAddressSpace : std::uint8_t {
    kNone,
    kDevice,
    kConstant,
    kThreadgroup,
    kThread,
};

enum class MslTypeKind : std::uint8_t {
    kVoid,
    kBool,
    kInt,
    kUInt,
    kHalf,
    kFloat,
    kDouble,
    kVector,
    kPointer,
    kReference,
    kStruct,
};

struct MslType {
    MslTypeKind kind = MslTypeKind::kVoid;
    std::uint32_t lanes = 1;
    MslAddressSpace address_space = MslAddressSpace::kNone;
    std::shared_ptr<MslType> element;
    std::string struct_name;

    static MslType void_type();
    static MslType boolean();
    static MslType sint(std::uint32_t bits = 32);
    static MslType uint(std::uint32_t bits = 32);
    static MslType floating(std::uint32_t bits = 32);
    static MslType vector(MslType element, std::uint32_t lanes);
    static MslType pointer(MslType pointee, MslAddressSpace address_space);
    static MslType reference(MslType referent, MslAddressSpace address_space);

    [[nodiscard]] std::string str() const;
    friend bool operator==(const MslType& left, const MslType& right);
};

struct MslExpression;
using MslExpr = std::shared_ptr<MslExpression>;

struct MslIdentifier {
    std::string name;
};

struct MslLiteral {
    std::string spelling;
};

struct MslUnary {
    std::string operation;
    MslExpr operand;
};

struct MslBinary {
    std::string operation;
    MslExpr left;
    MslExpr right;
};

struct MslCall {
    std::string callee;
    std::vector<MslExpr> arguments;
};

struct MslCast {
    MslType target;
    MslExpr operand;
    bool reinterpret = false;
};

struct MslSubscript {
    MslExpr base;
    MslExpr index;
};

struct MslMember {
    MslExpr base;
    std::string member;
};

struct MslConditional {
    MslExpr condition;
    MslExpr true_value;
    MslExpr false_value;
};

struct MslExpression {
    MslType type;
    std::variant<MslIdentifier, MslLiteral, MslUnary, MslBinary, MslCall,
                 MslCast, MslSubscript, MslMember, MslConditional>
        value;
    ir::SourceLocation location;

    static MslExpr identifier(std::string name, MslType type);
    static MslExpr literal(std::string spelling, MslType type);
    static MslExpr unary(std::string operation, MslExpr operand, MslType type);
    static MslExpr binary(std::string operation, MslExpr left, MslExpr right, MslType type);
    static MslExpr call(std::string callee, std::vector<MslExpr> arguments, MslType type);
    static MslExpr cast(MslType target, MslExpr operand, bool reinterpret = false);
    static MslExpr subscript(MslExpr base, MslExpr index, MslType type);
    static MslExpr member(MslExpr base, std::string member, MslType type);
    static MslExpr conditional(MslExpr condition, MslExpr true_value,
                               MslExpr false_value, MslType type);
};

struct MslStatement;
using MslStmt = std::shared_ptr<MslStatement>;

struct MslVariableDeclaration {
    MslType type;
    std::string name;
    std::optional<MslExpr> initializer;
    bool is_const = false;
};

struct MslAssignment {
    MslExpr target;
    MslExpr value;
};

struct MslExpressionStatement {
    MslExpr expression;
};

struct MslIf {
    MslExpr condition;
    std::vector<MslStmt> then_statements;
    std::vector<MslStmt> else_statements;
};

struct MslWhile {
    MslExpr condition;
    std::vector<MslStmt> statements;
};

struct MslReturn {
    std::optional<MslExpr> value;
};

struct MslStatement {
    std::variant<MslVariableDeclaration, MslAssignment, MslExpressionStatement,
                 MslIf, MslWhile, MslReturn>
        value;
    ir::SourceLocation location;

    static MslStmt variable(MslType type, std::string name,
                            std::optional<MslExpr> initializer = std::nullopt,
                            bool is_const = false);
    static MslStmt assignment(MslExpr target, MslExpr value);
    static MslStmt expression(MslExpr expression);
    static MslStmt if_statement(MslExpr condition, std::vector<MslStmt> then_statements,
                                std::vector<MslStmt> else_statements = {});
    static MslStmt while_statement(MslExpr condition, std::vector<MslStmt> statements);
    static MslStmt return_statement(std::optional<MslExpr> value = std::nullopt);
};

struct MslAttribute {
    std::string name;
    std::optional<std::uint32_t> index;
};

struct MslParameter {
    MslType type;
    std::string name;
    std::vector<MslAttribute> attributes;
};

struct MslFunction {
    std::string name;
    MslType return_type = MslType::void_type();
    bool is_kernel = false;
    std::vector<MslParameter> parameters;
    std::vector<MslStmt> statements;
};

struct MslStructField {
    MslType type;
    std::string name;
};

struct MslStruct {
    std::string name;
    std::vector<MslStructField> fields;
};

struct MslModule {
    std::uint32_t language_major = 3;
    std::uint32_t language_minor = 1;
    std::vector<std::string> comments;
    std::vector<std::string> includes = {"metal_stdlib"};
    std::vector<MslStruct> structs;
    std::vector<MslFunction> functions;
};

struct MslPrintResult {
    bool ok = false;
    std::string source;
    std::vector<std::string> errors;
};

[[nodiscard]] std::string sanitize_identifier(std::string_view name);
[[nodiscard]] MslPrintResult print_msl(const MslModule& module);

}  // namespace cumetal::metal
