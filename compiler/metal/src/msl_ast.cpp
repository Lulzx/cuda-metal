#include "cumetal/metal/msl_ast.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace cumetal::metal {
namespace {

std::string address_space_spelling(MslAddressSpace address_space) {
    switch (address_space) {
        case MslAddressSpace::kNone: return "";
        case MslAddressSpace::kDevice: return "device";
        case MslAddressSpace::kConstant: return "constant";
        case MslAddressSpace::kThreadgroup: return "threadgroup";
        case MslAddressSpace::kThread: return "thread";
    }
    return "";
}

int precedence(const MslExpression& expression) {
    if (std::holds_alternative<MslConditional>(expression.value)) return 2;
    if (const auto* binary = std::get_if<MslBinary>(&expression.value)) {
        const std::string& op = binary->operation;
        if (op == "||") return 3;
        if (op == "&&") return 4;
        if (op == "|") return 5;
        if (op == "^") return 6;
        if (op == "&") return 7;
        if (op == "==" || op == "!=") return 8;
        if (op == "<" || op == "<=" || op == ">" || op == ">=") return 9;
        if (op == "<<" || op == ">>") return 10;
        if (op == "+" || op == "-") return 11;
        if (op == "*" || op == "/" || op == "%") return 12;
    }
    if (std::holds_alternative<MslUnary>(expression.value) ||
        std::holds_alternative<MslCast>(expression.value)) {
        return 13;
    }
    return 14;
}

class Printer {
public:
    MslPrintResult run(const MslModule& module) {
        for (const std::string& comment : module.comments) {
            if (comment.find('\n') != std::string::npos ||
                comment.find('\r') != std::string::npos) {
                errors_.push_back("MSL metadata comments must be single-line");
                continue;
            }
            out_ << "// " << comment << "\n";
        }
        if (!module.comments.empty()) out_ << "\n";
        for (const std::string& include : module.includes) {
            if (include.empty()) {
                errors_.push_back("MSL include name cannot be empty");
                continue;
            }
            out_ << "#include <" << include << ">\n";
        }
        out_ << "using namespace metal;\n\n";

        std::unordered_set<std::string> names;
        for (const MslStruct& structure : module.structs) {
            const std::string name = sanitize_identifier(structure.name);
            if (!names.insert(name).second) {
                errors_.push_back("duplicate MSL declaration: " + name);
                continue;
            }
            out_ << "struct " << name << " {\n";
            for (const MslStructField& field : structure.fields) {
                out_ << "    " << field.type.str() << " " << sanitize_identifier(field.name) << ";\n";
            }
            out_ << "};\n\n";
        }

        for (const MslGlobalByteArray& global : module.global_byte_arrays) {
            const std::string name = sanitize_identifier(global.name);
            if (!names.insert(name).second) {
                errors_.push_back("duplicate MSL declaration: " + name);
                continue;
            }
            if (global.bytes.empty()) {
                errors_.push_back("global byte array cannot be empty: " + name);
                continue;
            }
            out_ << "constant uchar " << name << "[" << global.bytes.size()
                 << "] = {";
            for (std::size_t i = 0; i < global.bytes.size(); ++i) {
                if (i != 0) out_ << ", ";
                out_ << "0x" << std::hex << std::setw(2) << std::setfill('0')
                     << static_cast<unsigned>(global.bytes[i]) << std::dec;
            }
            out_ << "};\n\n";
        }

        for (const MslFunction& function : module.functions) {
            const std::string name = sanitize_identifier(function.name);
            if (!names.insert(name).second) {
                errors_.push_back("duplicate MSL declaration: " + name);
                continue;
            }
            if (function.is_kernel) {
                out_ << "kernel ";
            }
            out_ << function.return_type.str() << " " << name << "(\n";
            std::unordered_set<std::string> parameter_names;
            for (std::size_t i = 0; i < function.parameters.size(); ++i) {
                const MslParameter& parameter = function.parameters[i];
                const std::string parameter_name = sanitize_identifier(parameter.name);
                if (!parameter_names.insert(parameter_name).second) {
                    errors_.push_back("duplicate parameter '" + parameter_name +
                                      "' in function '" + name + "'");
                }
                out_ << "    " << parameter.type.str() << " " << parameter_name;
                for (const MslAttribute& attribute : parameter.attributes) {
                    out_ << " [[" << attribute.name;
                    if (attribute.index.has_value()) {
                        out_ << "(" << *attribute.index << ")";
                    }
                    out_ << "]]";
                }
                out_ << (i + 1 == function.parameters.size() ? "\n" : ",\n");
            }
            out_ << ") {\n";
            for (const MslStmt& statement : function.statements) {
                print_statement(statement, 1);
            }
            out_ << "}\n\n";
        }

        return {
            .ok = errors_.empty(),
            .source = out_.str(),
            .errors = std::move(errors_),
        };
    }

private:
    void indent(int depth) {
        for (int i = 0; i < depth; ++i) {
            out_ << "    ";
        }
    }

    void print_expression(const MslExpr& expression, int parent_precedence = 0) {
        if (expression == nullptr) {
            errors_.push_back("null MSL expression");
            out_ << "/* invalid */";
            return;
        }
        const int current_precedence = precedence(*expression);
        const bool parenthesize = current_precedence < parent_precedence;
        if (parenthesize) out_ << "(";

        std::visit(
            [&](const auto& node) {
                using Node = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<Node, MslIdentifier>) {
                    out_ << sanitize_identifier(node.name);
                } else if constexpr (std::is_same_v<Node, MslLiteral>) {
                    out_ << node.spelling;
                } else if constexpr (std::is_same_v<Node, MslUnary>) {
                    out_ << node.operation;
                    print_expression(node.operand, current_precedence);
                } else if constexpr (std::is_same_v<Node, MslBinary>) {
                    print_expression(node.left, current_precedence);
                    out_ << " " << node.operation << " ";
                    print_expression(node.right, current_precedence + 1);
                } else if constexpr (std::is_same_v<Node, MslCall>) {
                    out_ << sanitize_identifier(node.callee) << "(";
                    for (std::size_t i = 0; i < node.arguments.size(); ++i) {
                        if (i != 0) out_ << ", ";
                        print_expression(node.arguments[i]);
                    }
                    out_ << ")";
                } else if constexpr (std::is_same_v<Node, MslCast>) {
                    if (node.bitcast) {
                        out_ << "as_type<" << node.target.str() << ">(";
                        print_expression(node.operand);
                        out_ << ")";
                    } else if (node.reinterpret) {
                        out_ << "reinterpret_cast<" << node.target.str() << ">(";
                        print_expression(node.operand);
                        out_ << ")";
                    } else {
                        out_ << node.target.str() << "(";
                        print_expression(node.operand);
                        out_ << ")";
                    }
                } else if constexpr (std::is_same_v<Node, MslSubscript>) {
                    print_expression(node.base, current_precedence);
                    out_ << "[";
                    print_expression(node.index);
                    out_ << "]";
                } else if constexpr (std::is_same_v<Node, MslMember>) {
                    print_expression(node.base, current_precedence);
                    out_ << "." << sanitize_identifier(node.member);
                } else if constexpr (std::is_same_v<Node, MslConditional>) {
                    print_expression(node.condition, current_precedence);
                    out_ << " ? ";
                    print_expression(node.true_value, current_precedence);
                    out_ << " : ";
                    print_expression(node.false_value, current_precedence);
                } else if constexpr (std::is_same_v<Node, MslVoteMask>) {
                    out_ << "uint(simd_vote::vote_t(";
                    print_expression(node.vote);
                    out_ << "))";
                }
            },
            expression->value);

        if (parenthesize) out_ << ")";
    }

    void print_statement(const MslStmt& statement, int depth) {
        if (statement == nullptr) {
            errors_.push_back("null MSL statement");
            return;
        }
        std::visit(
            [&](const auto& node) {
                using Node = std::decay_t<decltype(node)>;
                if constexpr (std::is_same_v<Node, MslVariableDeclaration>) {
                    indent(depth);
                    out_ << node.type.str();
                    if (node.is_const) out_ << " const";
                    out_ << " " << sanitize_identifier(node.name);
                    if (node.initializer.has_value()) {
                        out_ << " = ";
                        print_expression(*node.initializer);
                    }
                    out_ << ";\n";
                } else if constexpr (std::is_same_v<Node, MslAssignment>) {
                    indent(depth);
                    print_expression(node.target);
                    out_ << " = ";
                    print_expression(node.value);
                    out_ << ";\n";
                } else if constexpr (std::is_same_v<Node, MslExpressionStatement>) {
                    indent(depth);
                    print_expression(node.expression);
                    out_ << ";\n";
                } else if constexpr (std::is_same_v<Node, MslIf>) {
                    indent(depth);
                    out_ << "if (";
                    print_expression(node.condition);
                    out_ << ") {\n";
                    for (const MslStmt& child : node.then_statements) print_statement(child, depth + 1);
                    indent(depth);
                    if (node.else_statements.empty()) {
                        out_ << "}\n";
                    } else {
                        out_ << "} else {\n";
                        for (const MslStmt& child : node.else_statements) print_statement(child, depth + 1);
                        indent(depth);
                        out_ << "}\n";
                    }
                } else if constexpr (std::is_same_v<Node, MslWhile>) {
                    indent(depth);
                    out_ << "while (";
                    print_expression(node.condition);
                    out_ << ") {\n";
                    for (const MslStmt& child : node.statements) print_statement(child, depth + 1);
                    indent(depth);
                    out_ << "}\n";
                } else if constexpr (std::is_same_v<Node, MslSwitch>) {
                    indent(depth);
                    out_ << "switch (";
                    print_expression(node.selector);
                    out_ << ") {\n";
                    for (const MslSwitchCase& switch_case : node.cases) {
                        indent(depth + 1);
                        out_ << "case ";
                        print_expression(switch_case.value);
                        out_ << ": {\n";
                        for (const MslStmt& child : switch_case.statements) {
                            print_statement(child, depth + 2);
                        }
                        indent(depth + 1);
                        out_ << "}\n";
                    }
                    indent(depth);
                    out_ << "}\n";
                } else if constexpr (std::is_same_v<Node, MslReturn>) {
                    indent(depth);
                    out_ << "return";
                    if (node.value.has_value()) {
                        out_ << " ";
                        print_expression(*node.value);
                    }
                    out_ << ";\n";
                } else if constexpr (std::is_same_v<Node, MslBreak>) {
                    indent(depth);
                    out_ << "break;\n";
                }
            },
            statement->value);
    }

    std::ostringstream out_;
    std::vector<std::string> errors_;
};

}  // namespace

MslType MslType::void_type() {
    return {};
}

MslType MslType::boolean() {
    return {.kind = MslTypeKind::kBool};
}

MslType MslType::sint(std::uint32_t bits) {
    return {.kind = bits <= 16 ? MslTypeKind::kInt : MslTypeKind::kInt, .lanes = bits};
}

MslType MslType::uint(std::uint32_t bits) {
    return {.kind = MslTypeKind::kUInt, .lanes = bits};
}

MslType MslType::floating(std::uint32_t bits) {
    if (bits == 16) return {.kind = MslTypeKind::kHalf};
    if (bits == 64) return {.kind = MslTypeKind::kDouble};
    return {.kind = MslTypeKind::kFloat};
}

MslType MslType::vector(MslType vector_element, std::uint32_t vector_lanes) {
    return {
        .kind = MslTypeKind::kVector,
        .lanes = vector_lanes,
        .element = std::make_shared<MslType>(std::move(vector_element)),
    };
}

MslType MslType::pointer(MslType pointee, MslAddressSpace pointer_address_space) {
    return {
        .kind = MslTypeKind::kPointer,
        .address_space = pointer_address_space,
        .element = std::make_shared<MslType>(std::move(pointee)),
    };
}

MslType MslType::reference(MslType referent, MslAddressSpace reference_address_space) {
    return {
        .kind = MslTypeKind::kReference,
        .address_space = reference_address_space,
        .element = std::make_shared<MslType>(std::move(referent)),
    };
}

bool operator==(const MslType& left, const MslType& right) {
    if (left.kind != right.kind || left.lanes != right.lanes ||
        left.address_space != right.address_space ||
        left.struct_name != right.struct_name) {
        return false;
    }
    if (left.element == nullptr || right.element == nullptr) {
        return left.element == nullptr && right.element == nullptr;
    }
    return *left.element == *right.element;
}

std::string MslType::str() const {
    switch (kind) {
        case MslTypeKind::kVoid: return "void";
        case MslTypeKind::kBool: return "bool";
        case MslTypeKind::kInt:
            if (lanes == 8) return "char";
            if (lanes == 16) return "short";
            if (lanes == 64) return "long";
            return "int";
        case MslTypeKind::kUInt:
            if (lanes == 8) return "uchar";
            if (lanes == 16) return "ushort";
            if (lanes == 64) return "ulong";
            return "uint";
        case MslTypeKind::kHalf: return "half";
        case MslTypeKind::kFloat: return "float";
        case MslTypeKind::kDouble: return "double";
        case MslTypeKind::kVector:
            return (element == nullptr ? std::string("uint") : element->str()) +
                   std::to_string(lanes);
        case MslTypeKind::kPointer:
        case MslTypeKind::kReference: {
            const std::string address_space = address_space_spelling(this->address_space);
            return (address_space.empty() ? std::string{} : address_space + " ") +
                   (element == nullptr ? std::string("void") : element->str()) +
                   (kind == MslTypeKind::kPointer ? "*" : "&");
        }
        case MslTypeKind::kStruct: return sanitize_identifier(struct_name);
    }
    return "void";
}

MslExpr MslExpression::identifier(std::string name, MslType type) {
    return std::make_shared<MslExpression>(
        MslExpression{.type = std::move(type), .value = MslIdentifier{std::move(name)}});
}

MslExpr MslExpression::literal(std::string spelling, MslType type) {
    return std::make_shared<MslExpression>(
        MslExpression{.type = std::move(type), .value = MslLiteral{std::move(spelling)}});
}

MslExpr MslExpression::unary(std::string operation, MslExpr operand, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslUnary{.operation = std::move(operation), .operand = std::move(operand)},
    });
}

MslExpr MslExpression::binary(std::string operation, MslExpr left, MslExpr right, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslBinary{
            .operation = std::move(operation),
            .left = std::move(left),
            .right = std::move(right),
        },
    });
}

MslExpr MslExpression::call(std::string callee, std::vector<MslExpr> arguments, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslCall{.callee = std::move(callee), .arguments = std::move(arguments)},
    });
}

MslExpr MslExpression::cast(MslType target, MslExpr operand, bool reinterpret) {
    const MslType result_type = target;
    return std::make_shared<MslExpression>(MslExpression{
        .type = result_type,
        .value = MslCast{
            .target = std::move(target),
            .operand = std::move(operand),
            .reinterpret = reinterpret,
        },
    });
}

MslExpr MslExpression::bitcast(MslType target, MslExpr operand) {
    const MslType result_type = target;
    return std::make_shared<MslExpression>(MslExpression{
        .type = result_type,
        .value = MslCast{
            .target = std::move(target),
            .operand = std::move(operand),
            .bitcast = true,
        },
    });
}

MslExpr MslExpression::subscript(MslExpr base, MslExpr index, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslSubscript{.base = std::move(base), .index = std::move(index)},
    });
}

MslExpr MslExpression::member(MslExpr base, std::string member_name, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslMember{.base = std::move(base), .member = std::move(member_name)},
    });
}

MslExpr MslExpression::conditional(MslExpr condition, MslExpr true_value,
                                   MslExpr false_value, MslType type) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = std::move(type),
        .value = MslConditional{
            .condition = std::move(condition),
            .true_value = std::move(true_value),
            .false_value = std::move(false_value),
        },
    });
}

MslExpr MslExpression::vote_mask(MslExpr vote) {
    return std::make_shared<MslExpression>(MslExpression{
        .type = MslType::uint(),
        .value = MslVoteMask{.vote = std::move(vote)},
    });
}

MslStmt MslStatement::variable(MslType type, std::string name,
                               std::optional<MslExpr> initializer, bool is_const) {
    return std::make_shared<MslStatement>(MslStatement{
        .value = MslVariableDeclaration{
            .type = std::move(type),
            .name = std::move(name),
            .initializer = std::move(initializer),
            .is_const = is_const,
        },
    });
}

MslStmt MslStatement::assignment(MslExpr target, MslExpr value) {
    return std::make_shared<MslStatement>(
        MslStatement{.value = MslAssignment{std::move(target), std::move(value)}});
}

MslStmt MslStatement::expression(MslExpr expression) {
    return std::make_shared<MslStatement>(
        MslStatement{.value = MslExpressionStatement{std::move(expression)}});
}

MslStmt MslStatement::if_statement(MslExpr condition, std::vector<MslStmt> then_statements,
                                   std::vector<MslStmt> else_statements) {
    return std::make_shared<MslStatement>(MslStatement{
        .value = MslIf{
            .condition = std::move(condition),
            .then_statements = std::move(then_statements),
            .else_statements = std::move(else_statements),
        },
    });
}

MslStmt MslStatement::while_statement(MslExpr condition, std::vector<MslStmt> statements) {
    return std::make_shared<MslStatement>(MslStatement{
        .value = MslWhile{
            .condition = std::move(condition),
            .statements = std::move(statements),
        },
    });
}

MslStmt MslStatement::switch_statement(MslExpr selector,
                                        std::vector<MslSwitchCase> cases) {
    return std::make_shared<MslStatement>(MslStatement{
        .value = MslSwitch{
            .selector = std::move(selector),
            .cases = std::move(cases),
        },
    });
}

MslStmt MslStatement::return_statement(std::optional<MslExpr> value) {
    return std::make_shared<MslStatement>(
        MslStatement{.value = MslReturn{.value = std::move(value)}});
}

MslStmt MslStatement::break_statement() {
    return std::make_shared<MslStatement>(MslStatement{.value = MslBreak{}});
}

std::string sanitize_identifier(std::string_view name) {
    static const std::unordered_set<std::string> kReserved = {
        "alignas", "alignof", "and", "asm", "atomic", "auto", "bool", "break", "case",
        "char", "class", "constant", "continue", "default", "device", "do", "double",
        "else", "enum", "false", "float", "for", "friend", "goto", "half", "if", "inline",
        "int", "kernel", "long", "namespace", "operator", "or", "private", "protected",
        "public", "register", "return", "short", "signed", "sizeof", "static", "struct",
        "switch", "template", "this", "thread", "threadgroup", "true", "typedef", "typename",
        "uint", "ulong", "union", "unsigned", "using", "virtual", "void", "volatile", "while",
    };

    std::string out;
    out.reserve(name.size() + 4);
    for (char c : name) {
        const unsigned char byte = static_cast<unsigned char>(c);
        out.push_back(std::isalnum(byte) != 0 || c == '_' ? c : '_');
    }
    if (out.empty()) out = "unnamed";
    if (std::isdigit(static_cast<unsigned char>(out.front())) != 0) out.insert(out.begin(), '_');
    if (kReserved.contains(out) || out.rfind("__", 0) == 0) out = "cm_" + out;
    return out;
}

MslPrintResult print_msl(const MslModule& module) {
    return Printer{}.run(module);
}

}  // namespace cumetal::metal
