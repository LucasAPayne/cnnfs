/* TODO(lucas): This is a quick and dirty parser implementation.
 * Below is the beginning of a list of considerations to eventually turn this into a "real" parser
 * and add some more useful metaprogramming features.
 * - Parser features
 *     - Abstract syntax tree
 *     - Actual recursive descent parsing
 * 
 * - Language features
 *     - Better and more consistent syntax across all areas (defining structs, functions, statements, etc.)
 *     - Get rid of ugly @api_iface/impl_begin/end tags and just have functions
 *     - Automatic handling of some things like putting in the #pragma once
 *       and putting, for example, #include "vector.h" at the top of "vector.c"
 *     - Error handling and error messages
 *     
 *     - Range-based for loops
 *     - Python-style f-strings
 *     - Eliminate the arrow operator
 *     - Nested functions
 */

#include "arena.h"
#include "log.h"
#include "parser.h"
#include "types.h"

#include <stdio.h>

#define MAX_PATH_LEN 256

// NOTE(lucas): Read an entire file into memory and add a null terminator at the end
internal char* read_file_into_memory(char* file_name, Arena* arena)
{
    char* result = 0;

    FILE* file = fopen(file_name, "r");
    if (file)
    {
        fseek(file, 0, SEEK_END);
        usize file_size = ftell(file);
        fseek(file, 0, SEEK_SET);

        result = push_array(arena, file_size+1, char);
        fread(result, file_size, 1, file);
        result[file_size] = 0; // Null-terminate file contents

        fclose(file);
    }
    else
    {
        log_fatal("File %s not found.\n", file_name);
    }

    return result;
}

Parser parser_init(char* file_path)
{
    Parser parser = {0};

    Arena arena = arena_alloc(MEGABYTES(10));
    parser.arena = arena;

    char* file_contents = read_file_into_memory(file_path, &parser.arena);
    ASSERT(file_contents);
    Tokenizer tokenizer = {0};
    tokenizer.at = file_contents;
    parser.tokenizer = tokenizer;

    return parser;
}

internal i64 abs_i64(i64 val)
{
    i64 result = val > 0 ? val : -val;
    return result;
}

//
// NOTE(lucas): String utils
//
// IMPORTANT(lucas): Strings are assumed to be null-terminated in these functions
// TODO(lucas): Optimize string methods
// TODO(lucas): Check for null strings as well as asserting
internal i64 str_len(char* s)
{
    // NOTE(lucas): To get length of a string, get another pointer,
    // then advance until the null terminator and get the difference
    // between the copy and original pointer location
    ASSERT(s);
    char* copy = s;
    for (copy; *copy; ++copy)
        ;
    i64 result = copy - s;
    return result;
}

internal char* str_dup(char* source, Arena* arena)
{
    ASSERT(source);
    usize len = str_len(source);
    char* result = push_array(arena, len+1, char);
    for (usize i = 0; i < len; ++i)
        result[i] = source[i];

    result[len] = '\0';
    return result;
}

internal char* str_cat(char* dest, char* suffix, Arena* arena)
{
    // NOTE(lucas): Assumes both strings are null terminated and
    // that dest has enough space to accommodate concatenation,
    // including the null terminator
    // NOTE(lucas): This operation is not done in place.
    // The returned string must be used
    ASSERT(dest);
    ASSERT(suffix);
    usize dest_len = str_len(dest);
    usize suffix_len = str_len(suffix);
    char* result = push_array(arena, dest_len+suffix_len+1, char);
    for (usize i = 0; i < dest_len; ++i)
        result[i] = dest[i];
    for (usize i = 0; i < suffix_len; ++i)
        result[dest_len+i] = suffix[i];

    result[dest_len+suffix_len] = '\0';
    return result;
}

// NOTE(lucas): Check if str has substring sub,
// returning a pointer to the first occurrence of sub
char* str_str(char *str, char *sub)
{
    const char* a = str;
    const char* b = sub;

    if (!a || !b) return 0;

    for (;;)
    {
        if (!*b)
            return (char *)str;
        if (!*a)
            return 0;
        if (*a++ != *b++)
        {
            a = ++str;
            b = sub;
        }
    }
}

inline char* str_sub(char* src, usize pos, usize len, Arena* arena)
{
    usize src_len = str_len(src);
    char* result = push_array(arena, len+1, char);
    
    if (pos >= src_len)
    {
        pos = len - 1;
        len = 0;
    }
    if (len > src_len)
        len = src_len;

    for (u32 i = 0; i < src_len; ++i, ++src)
        *(result + i) = *(src + pos);
    
    *(result + len) = '\0';
    return result;
}

// NOTE(lucas): Return a copy of a string from the beginning
// to the first occurrence of the substring
char* str_until(char* str, char* sub, Arena* arena)
{
    char* a = str;
    char* b = sub;
    usize len = 0;
    for (;;)
    {
        if (!*a)
            return 0;
        if (!*b)
        {
            char* result = str_sub(str, 0, len, arena);
            return result;
        }
        if (*a++ != *b++)
        {
            b = sub;
            ++len;
        }
    }
}

// NOTE(lucas): Check if str has substring sub,
// returning a pointer to the last occurrence of sub
char* str_after_last(char* str, char* sub)
{
    char* result = 0;
    i64 len = str_len(str);
    i64 sub_len = str_len(sub);

    i64 last_possible = str_len(str) - str_len(sub);
    char* p = str + last_possible;
    while (p >= str)
    {
        result = str_str(p, sub);
        if (result)
        {
            ++result;
            break;
        }
        
        --p;
    }

    return result;
}

// NOTE(lucas): In the string str, replace all occurrences of
// substring old_str with new_str
internal char* str_replace(char* str, char* old_str, char* new_str, Arena* arena)
{
    ASSERT(str);
    ASSERT(old_str);
    ASSERT(new_str);

    char* result = 0;

    i64 len = str_len(str);
    i64 old_len = str_len(old_str);
    i64 new_len = str_len(new_str);

    // Count number of occurrences of old in str
    usize occur = 0;
    for (i64 i = 0; i < len; ++i)
    {
        // Find a match, and if there is one, jump past it.
        char* match = str_str(&str[i], old_str);
        if (match)
        {
            ++occur;
            i += old_len - 1;
        }
    }

    if (occur > 0)
    {
        // The length of the resulting string is the length of the base string plus the
        // positive difference of the length in substrings, plus one for the null terminator.
        i64 result_len = len + occur*abs_i64(new_len - old_len) + 1;
        result = push_array(arena, result_len, char);

        i64 i = 0;
        while (*str)
        {
            // If the current index is the start of the substring
            if (str_str(str, old_str) == str)
            {
                for (i64 j = 0; j < new_len; ++j)
                    result[i+j] = new_str[j];
                i += new_len;
                str += old_len;
            }
            else
            {
                result[i++] = *str++;
            }
        }
        result[i] = '\0';
    }

    return result;
}

//
// NOTE(lucas): Token utils
//
internal Token token_create(char* text, TokenType type)
{
    Token result =
    {
        .text = text,
        .length = strlen(text),
        .type = type
    };
    return result;
}

internal Token token_copy(Token token, Arena* arena)
{
    char* dest = push_array(arena, token.length+1, char);
    char* text = token.text;
    for (usize i = 0; i < token.length; ++i)
        dest[i] = *text++;

    dest[token.length] = '\0';
    Token result = token_create(dest, token.type);
    return result;
}

internal Token token_append(Token token, Token suffix, Arena* arena)
{
    // TODO(lucas): It would probably be safer if this function handled
    // copying and null-terminating the suffix.
    // Take suffix as a char*?

    Token result = token_copy(token, arena);
    result.text = str_cat(result.text, suffix.text, arena);
    result.length = str_len(result.text);
    return result;
}

internal Token token_replace(Token token, char* old_str, char* new_str, Arena* arena)
{
    Token result = token_copy(token, arena);
    char* new_text = str_replace(result.text, old_str, new_str, arena);
    if (new_text)
    {
        result.text = new_text;
        result.length = str_len(result.text);
    }

    return result;
}

// TODO(lucas): Probably should name these functions with char rather than token
inline b32 token_is_end_of_line(char c)
{
    b32 result = (c == '\n') || (c == '\r');
    return result;
}

inline b32 token_is_whitespace(char c)
{
    // b32 result = (c == ' ' ) || (c == '\t') || (c == '\v') || (c == '\f') || token_is_end_of_line(c);
    b32 result = (c == ' ' ) || (c == '\t') || (c == '\v') || (c == '\f');
    return result;
}

inline b32 token_is_alpha(char c)
{
    b32 result = (((c >= 'a') && (c <= 'z')) ||
                  ((c >= 'A') && (c <= 'Z')));
    return result;
}

inline b32 token_is_digit(char c)
{
    b32 result = ((c >= '0') && (c <= '9'));
    return result;
}

inline b32 token_equals(Token token, char* match)
{
    char* at = match;
    for (int i = 0; i < token.length; ++i, ++at)
    {
        if (!*at || (token.text[i] != *at))
            return false;
    }

    b32 result = (*at == 0);
    return result;
}

internal void tokenizer_eat_whitespace(Tokenizer* tokenizer)
{
    // NOTE(lucas): Eliminate whitesapce
    for (;;)
    {
        if (token_is_whitespace(tokenizer->at[0]))
            ++tokenizer->at;
        else if ((tokenizer->at[0] == '/') && (tokenizer->at[1] == '/'))
        {
            tokenizer->at += 2;
            while (!token_is_end_of_line(tokenizer->at[0]))
                ++tokenizer->at;
        }
        else if (tokenizer->at[0] && (tokenizer->at[0] == '/') && (tokenizer->at[1] == '*'))
        {
            tokenizer->at += 2;
            while (tokenizer->at[0] && !((tokenizer->at[0] == '*') && tokenizer->at[1] == '/'))
                ++tokenizer->at;
            
            // NOTE(lucas): Skip past the closing "*/"
            if (tokenizer->at[0] == '*')
                tokenizer->at += 2;
        }
        else break;
    }
}

internal Token token_get(Parser* parser)
{
    Tokenizer* tokenizer = &parser->tokenizer;
    tokenizer_eat_whitespace(tokenizer);

    Token token = {0};
    token.text = tokenizer->at;
    switch (tokenizer->at[0])
    {
        case '.':  token.type = TOK_DOT;       ++tokenizer->at; break;
        case ',':  token.type = TOK_COMMA;     ++tokenizer->at; break;
        case ':':  token.type = TOK_COLON;     ++tokenizer->at; break;
        case ';':  token.type = TOK_SEMICOLON; ++tokenizer->at; break;
        case '*':  token.type = TOK_STAR;      ++tokenizer->at; break;
        case '?':  token.type = TOK_QUESTION;  ++tokenizer->at; break;
        case '#':  token.type = TOK_HASH;      ++tokenizer->at; break;
        case '~':  token.type = TOK_BIT_NOT;   ++tokenizer->at; break;
        case '(':  token.type = TOK_LPAREN;    ++tokenizer->at; break;
        case ')':  token.type = TOK_RPAREN;    ++tokenizer->at; break;
        case '[':  token.type = TOK_LBRACKET;  ++tokenizer->at; break;
        case ']':  token.type = TOK_RBRACKET;  ++tokenizer->at; break;
        case '{':  token.type = TOK_LBRACE;    ++tokenizer->at; break;
        case '}':  token.type = TOK_RBRACE;    ++tokenizer->at; break;
        case '\n': token.type = TOK_EOL;       ++tokenizer->at; break;
        case '\r': token.type = TOK_EOL;       ++tokenizer->at; break;
        case '\0': token.type = TOK_EOF;       ++tokenizer->at; break;

        case '+':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '+')
            {
                token.type = TOK_INCREMENT;
                ++tokenizer->at;
            }
            else if (*tokenizer->at == '=')
            {
                token.type = TOK_PLUS_EQ;
                ++tokenizer->at;
            }
            else
                token.type = TOK_PLUS;
        } break;

        case '-':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '-')
            {
                token.type = TOK_DECREMENT;
                ++tokenizer->at;
            }
            else if (*tokenizer->at == '=')
            {
                token.type = TOK_MINS_EQ;
            }
            else if (*tokenizer->at == '>')
            {
                token.type = TOK_ARROW;
                ++tokenizer->at;
            }
            else
                token.type = TOK_MINUS;
        } break;

        case '<':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '<')
            {
                ++tokenizer->at;
                if (*tokenizer->at == '=')
                {
                    token.type = TOK_LSHIFT_EQ;
                    ++tokenizer->at;
                }
                else
                    token.type = TOK_LSHIFT;
            }
            else if (*tokenizer->at == '=')
            {
                token.type = TOK_LE;
                ++tokenizer->at;
            }
            else
                token.type = TOK_LT;
        } break;

        case '>':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '>')
            {
                ++tokenizer->at;
                if (*tokenizer->at == '=')
                {
                    token.type = TOK_RSHIFT_EQ;
                    ++tokenizer->at;
                }
                else
                    token.type = TOK_RSHIFT;
            }
            else if (*tokenizer->at == '=')
            {
                token.type = TOK_GE;
                ++tokenizer->at;
            }
            else
                token.type = TOK_GT;
        } break;

        case '/':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_DIV_EQ;
                ++tokenizer->at;
            }
            else
                token.type = TOK_DIV;
        } break;

        case '%':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_MOD_EQ;
                ++tokenizer->at;
            }
            else
                token.type = TOK_MOD;
        } break;

        case '=':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_EQ;
                ++tokenizer->at;
            }
            else
                token.type = TOK_ASSIGN;
        } break;

        case '!':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_NE;
                ++tokenizer->at;
            }
            else
                token.type = TOK_NOT;
        } break;

        case '&':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_BIT_AND_EQ;
                ++tokenizer->at;
            }
            else if (*tokenizer->at == '&')
            {
                token.type = TOK_AND;
                ++tokenizer->at;
            }
            else
                token.type = TOK_BIT_AND;
        } break;

        case '|':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_BIT_OR_EQ;
                ++tokenizer->at;
            }
            else if (*tokenizer->at == '|')
            {
                token.type = TOK_OR;
                ++tokenizer->at;
            }
            else
                token.type = TOK_BIT_OR;
        }

        case '^':
        {
            ++tokenizer->at;
            if (*tokenizer->at == '=')
            {
                token.type = TOK_BIT_XOR_EQ;
                ++tokenizer->at;
            }
            else
                token.type = TOK_BIT_XOR;
        } break;

        // TODO(lucas): comma-separated tags for things like keywords in front of functions
        case '@':
        {
            token.type = TOK_TAG;

            // NOTE(lucas): Tags are formatted as @name(param)
            // We only want "name", then "param" will be parsed separately
            ++tokenizer->at;
            token.text = tokenizer->at;
            while (tokenizer->at[0] && tokenizer->at[0] != '(' &&
                   tokenizer->at[0] != '\n' && !token_is_whitespace(tokenizer->at[0]))
            {
                ++tokenizer->at;
            }
        } break;

        case '"':
        {
            token.type = TOK_STRING;

            token.text = tokenizer->at;
            ++tokenizer->at;
            while (tokenizer->at[0] && tokenizer->at[0] != '"')
            {
                if (tokenizer->at[0] == '\\' && tokenizer->at[1])
                    ++tokenizer->at;
                
                ++tokenizer->at;
            }

            // NOTE(lucas): Skip over closing quote
            ++tokenizer->at;
        } break;

        default:
        {
            if (token_is_alpha(tokenizer->at[0]))
            {
                token.type = TOK_IDENTIFIER;
                token.text = tokenizer->at;

                // NOTE(lucas): An identifier can contain letters, digits, and underscores
                // TODO(lucas): For now, an identifier can have an asterisk for pointers.
                while (token_is_alpha(tokenizer->at[0]) || token_is_digit(tokenizer->at[0]) ||
                       tokenizer->at[0] == '_' || tokenizer->at[0] == '*')
                    {
                        ++tokenizer->at;
                    }
            }
            else
            {
                token.type = TOK_UNKNOWN;
                ++tokenizer->at;
            }
        } break;
    }

    token.length = tokenizer->at - token.text;
    return token;
}

internal b32 token_require(Parser* parser, TokenType desired_type)
{
    Token token = token_get(parser);
    while (token.type == TOK_EOL)
        token = token_get(parser);

    b32 result = (token.type == desired_type);
    return result;
}

internal Tag parse_array_tag_params(Parser* parser, Token tag_name_token)
{
    Tag tag = {0};
    tag.name = tag_name_token;
    b32 parsing = true;
    while (parsing)
    {
        Token token = token_get(parser);
        switch(token.type)
        {
            case TOK_IDENTIFIER:
            {
                if (token_equals(tag.name, "array"))
                    tag.type = TAG_ARRAY;
                else
                    tag.type = TAG_UNKNOWN;
            } break;

            case TOK_RPAREN:
            case TOK_EOF:
            {
                parsing = false;
            } break;

            default: break;
        }
    }

    return tag;
}

internal Directive parse_directive(Parser* parser)
{
    Directive result = {0};
    result.token = token_create("#", TOK_IDENTIFIER);

    b32 parsing = true;
    b32 first_token = true;
    while(parsing)
    {
        // TODO(lucas): For macro definitions, consider that a slash allows continuing on the next line.

        Token token = token_get(parser);
        switch (token.type)
        {
            // NOTE(lucas): #include "foo.h" will be recognized as a string
            case TOK_IDENTIFIER:
            case TOK_STRING:
            {
                if (first_token) // #pragma, #include, #define, etc.
                {
                    Token suffix = token_copy(token, &parser->arena);
                    result.token = token_append(result.token, suffix, &parser->arena);
                    first_token = false;
                }
                else // The directive arguments
                {
                    Token space = token_create(" ", TOK_IDENTIFIER);
                    result.token = token_append(result.token, space, &parser->arena);
                    Token suffix = token_copy(token, &parser->arena);
                    result.token = token_append(result.token, suffix, &parser->arena);
                }
            } break;

            // NOTE(lucas): Some #include statements use <> instead of ""
            case TOK_LT:
            {
                // NOTE(lucas): First, append a space and the chevron
                Token space = token_create(" ", TOK_IDENTIFIER);
                result.token = token_append(result.token, space, &parser->arena);
                Token suffix = token_copy(token, &parser->arena);
                result.token = token_append(result.token, suffix, &parser->arena);

                // NOTE(lucas): If the token is <, keep capturing until >
                b32 parsing_chevron = true;
                while (parsing_chevron)
                {
                    Token new_token = token_copy(token_get(parser), &parser->arena);
                    result.token = token_append(result.token, new_token, &parser->arena);
                    if (new_token.type == TOK_GT)
                        parsing_chevron = false;
                }
            } break;

            case TOK_EOL:
            case TOK_EOF:
            {
                parsing = false;
            } break;
        }
    }

    return result;
}

internal Member parse_member(Parser* parser, Token member_name_token)
{
    Member member = {0};
    member.name = member_name_token;

    b32 parsing = true;
    while (parsing)
    {
        Token token = token_get(parser);
        switch (token.type)
        {
            case TOK_TAG:
            {
                member.tag = parse_array_tag_params(parser, token);
            } break;

            // NOTE(lucas): Tags should catch everything else legal,
            // so the only identifier should be the type
            case TOK_IDENTIFIER:
            {
                member.data_type = token;
            } break;

            case TOK_SEMICOLON:
            case TOK_EOF:
            {
                parsing = false;
            } break;

            default: break;
        }
    }

    return member;
}

internal Struct parse_struct(Parser* parser)
{
    Struct structure = {0};

    Token struct_token = token_get(parser);
    if (token_require(parser, TOK_LBRACE))
    {
        structure.name = struct_token;
        usize member_index = 0;
        b32 parsing = true;
        while (parsing)
        {
            Token member_name_token = token_get(parser);
            switch(member_name_token.type)
            {
                case TOK_IDENTIFIER:
                {
                    Member* member = push_struct(&parser->arena, Member);
                    *member = parse_member(parser, member_name_token);
                    list_push(&structure.member_list, member, &parser->arena);

                    if (token_equals(member->data_type, "numeric"))
                    {
                        structure.numeric = true;
                        structure.numeric_index = member_index;
                    }
                    ++member_index;
                } break;

                case TOK_RBRACE:
                case TOK_EOF:
                {
                    parsing = false;
                } break;

                default: break;
            }
        }
    }

    return structure;
}

internal Param parse_param(Parser* parser, Token token)
{
    Param param = {0};
    param.data_type = token;
    Token name_token = token_get(parser);
    if (name_token.type != TOK_EOF)
        param.name = name_token;

    return param;
}

internal b32 parse_keyword(Parser* parser, Token* keywords, Token keyword_token, char* keyword_str, b32 first_keyword)
{
    Token space = token_create(" ", TOK_IDENTIFIER);
    if (token_equals(keyword_token, keyword_str))
    {
        if (first_keyword)
            first_keyword = false;
        else
            *keywords = token_append(*keywords, space, &parser->arena);

        Token keyword = token_create(keyword_str, TOK_IDENTIFIER);;
        *keywords = token_append(*keywords, keyword, &parser->arena);
    }
    return first_keyword;
}

internal FunctionDecl parse_function_decl(Parser* parser, Token token)
{
    FunctionDecl decl = {0};

    b32 first_keyword = true;
    b32 first_id = true;

    decl.keywords.type = TOK_IDENTIFIER;
    if (token.type == TOK_TAG)
    {
        first_keyword = parse_keyword(parser, &decl.keywords, token, "inline", first_keyword);
        first_keyword = parse_keyword(parser, &decl.keywords, token, "internal", first_keyword);
    }
    else if (token.type == TOK_IDENTIFIER)
    {
        decl.return_type = token;
        decl.name = token_get(parser);
        first_id = false;
    }

    b32 parsing = true;
    while (parsing)
    {
        Token new_token = token_get(parser);
        switch (new_token.type)
        {
            case TOK_TAG:
            {
                first_keyword = parse_keyword(parser, &decl.keywords, new_token, "inline", first_keyword);
                first_keyword = parse_keyword(parser, &decl.keywords, new_token, "internal", first_keyword);
            } break;

            case TOK_IDENTIFIER:
            {
                if (first_id)
                {
                    decl.return_type = new_token;
                    decl.name = token_get(parser);
                    first_id = false;
                }
                else
                {
                    Param* param = push_struct(&parser->arena, Param);
                    *param = parse_param(parser, new_token);
                    list_push(&decl.param_list, param, &parser->arena);
                }
            } break;

            case TOK_RPAREN:
            case TOK_EOF:
            {
                parsing = false;
            } break;

            default: break;
        }
    }

    return decl;
}

internal FunctionDef parse_function_def(Parser* parser, Token token)
{
    FunctionDef def = {0};

    // NOTE(lucas): We can parse the function header the same way as the declaration
    FunctionDecl decl = parse_function_decl(parser, token);
    def.name = decl.name;
    def.return_type = decl.return_type;
    def.param_list = decl.param_list;
    def.keywords = decl.keywords;

    i32 nest_count = 0;
    b32 first_keyword = true;
    b32 parsing = true;
    while(parsing)
    {
        Token* new_token = push_struct(&parser->arena, Token);
        *new_token = token_get(parser);
        switch (new_token->type)
        {
            case TOK_LBRACE:
            {
                ++nest_count;
                list_push(&def.token_list, new_token, &parser->arena);
            } break;

            case TOK_RBRACE:
            {
                --nest_count;
                list_push(&def.token_list, new_token, &parser->arena);
                if (nest_count == 0)
                    parsing = false;
            } break;

            case TOK_EOF:
            {
                parsing = false;
            } break;

            default:
            {
                list_push(&def.token_list, new_token, &parser->arena);
            } break;
        }
    }

    return def;
}

internal Iface parse_iface(Parser* parser)
{
    // TODO(lucas): Error checking. Currently assuming no syntax errors.
    Iface iface = {0};

    b32 parsing = true;
    while (parsing)
    {
        Token token = token_get(parser);
        switch (token.type)
        {
            case TOK_TAG:
            {
                if (token_equals(token, "api_iface_end"))
                    parsing = false;
                else
                {
                    FunctionDecl* decl = push_struct(&parser->arena, FunctionDecl);
                    *decl = parse_function_decl(parser, token);
                    list_push(&iface.decl_list, decl, &parser->arena);
                }
            } break;

            case TOK_IDENTIFIER:
            {
                FunctionDecl* decl = push_struct(&parser->arena, FunctionDecl);
                *decl = parse_function_decl(parser, token);
                list_push(&iface.decl_list, decl, &parser->arena);
            } break;

            case TOK_EOF:
            {
                parsing = false;
            } break;

            default: break;
        }
    }

    return iface;
}

internal Impl parse_impl(Parser* parser)
{
    // TODO(lucas): Error checking. Currently assuming no syntax errors.
    Impl impl = {0};
    
    b32 parsing = true;
    while (parsing)
    {
        Token token = token_get(parser);
        switch (token.type)
        {
            case TOK_TAG:
            {
                // TODO(lucas): Remove extra newlines between closing brace and this tag
                if (token_equals(token, "api_impl_end"))
                    parsing = false;
                else
                {
                    FunctionDef* def = push_struct(&parser->arena, FunctionDef);
                    *def = parse_function_def(parser, token);
                    list_push(&impl.def_list, def, &parser->arena);
                }
            } break;

            case TOK_IDENTIFIER:
            {
                FunctionDef* def = push_struct(&parser->arena, FunctionDef);
                *def = parse_function_def(parser, token);
                list_push(&impl.def_list, def, &parser->arena);
            } break;

            case TOK_EOF:
            {
                parsing = false;
            } break;

            default: break;
        }
    }

    return impl;
}

internal void struct_write_to_file(FILE* file, Struct structure)
{
    if (structure.written_to_file)
        return;

    fprintf(file, "typedef struct %.*s\n", (int)structure.name.length, structure.name.text);
    fprintf(file, "{\n");

    list_foreach(node, structure.member_list)
    {
        Member* member = (Member*)node->data;
        if (token_equals(member->tag.name, "array"))
        {
            fprintf(file, "\t%.*s* %.*s;\n", (int)member->data_type.length, member->data_type.text,
                    (int)member->name.length, member->name.text);
        }
        else
        {
            fprintf(file, "\t%.*s %.*s;\n", (int)member->data_type.length, member->data_type.text,
                    (int)member->name.length, member->name.text);
        }
    }

    fprintf(file, "} %.*s;\n", (int)structure.name.length, structure.name.text);
}

internal void function_decl_write_to_file(FILE* file, FunctionDecl* decl)
{
    fprintf(file, "%.*s", (int)decl->keywords.length, decl->keywords.text);
    if (decl->keywords.text)
        fprintf(file, " ");

    fprintf(file, "%.*s ", (int)decl->return_type.length, decl->return_type.text);
    fprintf(file, "%.*s(", (int)decl->name.length, decl->name.text);

    // TODO: Pull out writing declaration to file into a function,
    // and search for a numeric data type like the structs do.
    list_foreach(node, decl->param_list)
    {
        Param* param = (Param*)node->data;
        fprintf(file, "%.*s ", (int)param->data_type.length, param->data_type.text);
        fprintf(file, "%.*s", (int)param->name.length, param->name.text);
        if (node->next)
            fprintf(file, ", ");
        else
            fprintf(file, ");\n");
    }
}

internal void function_def_write_to_file(FILE* file, FunctionDef* def)
{
    fprintf(file, "\n");

    fprintf(file, "%.*s", (int)def->keywords.length, def->keywords.text);
    if (def->keywords.text)
        fprintf(file, " ");

    fprintf(file, "%.*s ", (int)def->return_type.length, def->return_type.text);
    fprintf(file, "%.*s(", (int)def->name.length, def->name.text);

    // TODO: Pull out writing declaration to file into a function,
    // and search for a numeric data type like the structs do.
    list_foreach(node, def->param_list)
    {
        Param* param = (Param*)node->data;
        fprintf(file, "%.*s ", (int)param->data_type.length, param->data_type.text);
        fprintf(file, "%.*s", (int)param->name.length, param->name.text);
        if (node->next)
            fprintf(file, ", ");
        else
            fprintf(file, ")");
    }

    // TODO(lucas): Flags?
    b32 control_flow_keyword = false;
    b32 semicolon_encountered = false;

    i32 nest_count = 0;
    list_foreach(node, def->token_list)
    {
        Token* token = (Token*)node->data;
        fprintf(file, "%.*s", (int)token->length, token->text);

        // NOTE(lucas): Prevent adding an extra space in several circumstances
        // TODO(lucas): Prevent space in literal numbers, e.g., 0.0f
        b32 add_space = true;
        Token* next_token = 0;
        if (node->next)
            next_token = (Token*)node->next->data;
        switch (token->type)
        {
            case TOK_SEMICOLON:
            {
                if (next_token && next_token->type == TOK_EOL)
                    semicolon_encountered = true;
            } break;

            case TOK_LPAREN:
            case TOK_LBRACKET:
            case TOK_DOT:
            case TOK_ARROW:
            case TOK_INCREMENT:
            case TOK_DECREMENT:
            {
                add_space = false;
            } break;

            case TOK_LBRACE:
            {
                add_space = false;
                ++nest_count;
            } break;

            case TOK_RBRACE:
            {
                add_space = false;
                --nest_count;
                if (nest_count == 0)
                {
                    fprintf(file, "\n");
                }
            } break;

            case TOK_IDENTIFIER:
            {
                // TODO(lucas): do while statements will break this.
                // Just need to keep track of whether there has been a "do" keyword
                if (token_equals(*token, "if") || token_equals(*token, "else") ||
                    token_equals(*token, "for") || token_equals(*token, "while"))
                {
                        control_flow_keyword = true;
                }

                if (next_token)
                {
                    if (next_token->type == TOK_LPAREN    || next_token->type == TOK_ARROW     ||
                        next_token->type == TOK_COMMA     || next_token->type == TOK_INCREMENT ||
                        next_token->type == TOK_DECREMENT)
                    {
                        add_space = false;
                    }
                }
            } break;

            case TOK_EOL:
            {
                add_space = false;
                b32 indent_next_line = false;

                if (next_token)
                {
                    if (control_flow_keyword && next_token->type != TOK_LBRACE && !semicolon_encountered)
                    {
                        indent_next_line = true;   
                        ++nest_count;
                    }

                    for (i32 i = 0; i < nest_count; ++i)
                    {
                        // NOTE(lucas): Parsing hasn't hit the right brace yet to decrease nest count
                        if (next_token->type == TOK_RBRACE && i == nest_count-1)
                            break;

                        fprintf(file, "\t");
                    }
                }

                control_flow_keyword = false;
                semicolon_encountered = false;
                if (indent_next_line)
                    --nest_count;
            } break;

            default: break;
        }

        if (!next_token) continue;
        switch(next_token->type)
        {
            case TOK_LBRACKET:
            case TOK_RPAREN:
            case TOK_RBRACKET:
            case TOK_RBRACE:
            case TOK_SEMICOLON:
            case TOK_DOT:
            case TOK_EOL:
            {
                add_space = false;
            } break;
        }

        if (nest_count == 0)
            add_space = false;

        if (add_space)
            fprintf(file, " ");
    }
}

internal void write_output_file(char* file_name, Program* program, Arena* arena)
{
    char* h_file = str_cat(file_name, ".h", arena);
    char* c_file = str_cat(file_name, ".c", arena);

    // TODO(lucas): Is there a better way to store this or process each data type?
    Token numeric_data_types[] =
    {
        token_create("", TOK_UNKNOWN),
        token_create("u8",  TOK_IDENTIFIER),
        token_create("u16", TOK_IDENTIFIER),
        token_create("u32", TOK_IDENTIFIER),
        token_create("u64", TOK_IDENTIFIER),
        token_create("i8",  TOK_IDENTIFIER),
        token_create("i16", TOK_IDENTIFIER),
        token_create("i32", TOK_IDENTIFIER),
        token_create("i64", TOK_IDENTIFIER),
        token_create("f32", TOK_IDENTIFIER),
        token_create("f64", TOK_IDENTIFIER),
    };

    // NOTE(lucas): Used to store the name of a struct containing a numeric member
    // so that the function declarations and definitions can rename the struct name
    // in return types, parameters, and statements the same way that the structs do.
    // Token numeric_struct_name = {0};
    Token* numeric_struct_names = {0};

    FILE* out_h = fopen(h_file, "w");
    FILE* out_c = fopen(c_file, "w");
    if (h_file)
    {
        // TODO(lucas): Is there ever a situation where this is unwanted?
        fprintf(out_h, "#pragma once\n\n");

        list_foreach(node, program->directive_list)
        {
            Directive* directive = (Directive*)node->data;
            // TODO(lucas): Separate #include "", #include <>, #define, etc. by extra newline,
            // but group together the same kinds of directives
            fprintf(out_h, "%.*s\n\n", (int)directive->token.length, directive->token.text);
        }

        // NOTE(lucas): If a struct contains a numeric member, write separate structs for each numeric
        // data type. Otherwise, simply write the struct
        list_foreach(node, program->struct_list)
        {
            Struct* structure = (Struct*)node->data;
            if (structure->numeric)
            {
                for (NumericType num_type = 1; num_type < NUM_TYPE_COUNT; ++num_type)
                {
                    Struct new_struct = *structure;

                    Member* numeric_member = (Member*)list_at(structure->member_list, structure->numeric_index);
                    numeric_member->data_type = numeric_data_types[num_type];

                    usize suffix_length = numeric_data_types[num_type].length+2;
                    Token struct_suffix = token_create("_", TOK_IDENTIFIER);
                    struct_suffix = token_append(struct_suffix, numeric_data_types[num_type], arena);
                    new_struct.name = token_append(new_struct.name, struct_suffix, arena);

                    struct_write_to_file(out_h, new_struct);

                    if ((!structure->written_to_file) && (num_type < (NUM_TYPE_COUNT - 1)))
                        fprintf(out_h, "\n");
                }
            }
            else
                struct_write_to_file(out_h, *structure);
            
            structure->written_to_file = true;
        }

        fprintf(out_h, "\n");

        // TODO(lucas): There must be a better, simpler way than what I'm doing here.
        // TODO(lucas): All of these copies that can't get freed are very wasteful since several get created in the
        // inner loops

        // NOTE(lucas): If there is a numeric struct or type in function declarations,
        // write a separate declaration for each numeric type
        Token underscore = token_create("_", TOK_IDENTIFIER);
        Token numeric = token_create("numeric", TOK_IDENTIFIER);
        list_foreach(decl_node, program->iface.decl_list)
        {
            FunctionDecl* decl = (FunctionDecl*)decl_node->data;
            FunctionDecl new_decl = *decl;
            b32 numeric_found = false;
            for (NumericType num_type = 1; num_type < NUM_TYPE_COUNT; ++num_type)
            {
                list_foreach(struct_node, program->struct_list)
                {
                    Struct* structure = (Struct*)struct_node->data;
                    if (structure->numeric)
                    {
                        numeric_found = true;
                        // NOTE(lucas): See if function name contains struct name, and
                        // if so, replace it as with the struct names
                        Token function_name = token_copy(new_decl.name, arena);
                        Token numeric_struct_name = token_copy(structure->name, arena); // null-terminated copy
                        Token new_name = token_copy(structure->name, arena);
                        if (str_str(function_name.text, numeric_struct_name.text))
                        {
                            new_name = token_append(new_name, underscore, arena);
                            new_name = token_append(new_name, numeric_data_types[num_type], arena);
                            Token return_type = token_replace(decl->return_type, numeric_struct_name.text, new_name.text, arena);
                            if (!token_equals(new_decl.return_type, return_type.text))
                                new_decl.return_type = return_type;
                            else
                                new_decl.return_type = token_replace(decl->return_type, numeric.text, numeric_data_types[num_type].text, arena);

                            new_decl.name = token_replace(decl->name, numeric_struct_name.text, new_name.text, arena);
                        }
                    }
                }

                // NOTE(lucas): Check if any parameters contain the numeric struct or keyword
                // and replace them appropriately
                new_decl.param_list = list_copy(decl->param_list, sizeof(Param), arena);
                list_foreach(param_node, new_decl.param_list)
                {
                    Param* param = (Param*)param_node->data;
                    list_foreach(struct_node, program->struct_list)
                    {
                        Struct* structure = (Struct*)struct_node->data;
                        if (structure->numeric)
                        {
                            Token new_name = token_copy(structure->name, arena);
                            Token numeric_struct_name = token_copy(structure->name, arena);
                            if (str_str(param->data_type.text, numeric_struct_name.text))
                            {
                                new_name = token_append(new_name, underscore, arena);
                                new_name = token_append(new_name, numeric_data_types[num_type], arena);
                                Token param_data_type = token_replace(param->data_type, numeric_struct_name.text, new_name.text, arena);
                                if (!token_equals(param->data_type, param_data_type.text))
                                    param->data_type = param_data_type;
                                else
                                    param->data_type = token_replace(param->data_type, numeric.text, numeric_data_types[num_type].text, arena);

                                param->name = token_replace(param->name, numeric_struct_name.text, new_name.text, arena);
                            }
                        }
                    }
                }
                function_decl_write_to_file(out_h, &new_decl);
                if (num_type == (NUM_TYPE_COUNT - 1))
                    fprintf(out_h, "\n");
            }
            if (!numeric_found)
                function_decl_write_to_file(out_h, decl);
        }

        char* h_file_name = str_after_last(h_file, "/");
        fprintf(out_c, "#include \"%s\"\n", h_file_name);

        // NOTE(lucas): If there is a numeric struct or type in function definitions,
        // write a separate definition for each numeric type
        list_foreach(def_node, program->impl.def_list)
        {
            FunctionDef* def = (FunctionDef*)def_node->data;
            b32 numeric_found = false;
            b32 def_inline = (str_str(def->keywords.text, "inline") != 0);
            FILE* dest = def_inline ? out_h : out_c;
            for (NumericType num_type = 1; num_type < NUM_TYPE_COUNT; ++num_type)
            {
                FunctionDef new_def = *def;
                list_foreach(node, program->struct_list)
                {
                    Struct* structure = (Struct*)node->data;
                    if (structure->numeric)
                    {
                        numeric_found = true;
                        // TODO(lucas): Can copies be freed?
                        Token new_name = token_copy(structure->name, arena);
                        // NOTE(lucas): See if function name contains struct name, and
                        // if so, replace it as with the struct names
                        Token function_name = token_copy(new_def.name, arena);
                        Token numeric_struct_name = token_copy(structure->name, arena);
                        if (str_str(function_name.text, numeric_struct_name.text))
                        {
                            new_name = token_append(new_name, underscore, arena);
                            new_name = token_append(new_name, numeric_data_types[num_type], arena);
                            Token return_type = token_replace(def->return_type, numeric_struct_name.text, new_name.text, arena);
                            if (!token_equals(new_def.return_type, return_type.text))
                                new_def.return_type = return_type;
                            else    
                                new_def.return_type = token_replace(def->return_type, numeric.text, numeric_data_types[num_type].text, arena);
                            new_def.name = token_replace(def->name, numeric_struct_name.text, new_name.text, arena);
                        }
                    }
                }

                // NOTE(lucas): Check if any parameters contain the numeric struct or keyword
                // and replace them appropriately
                new_def.param_list = list_copy(def->param_list, sizeof(Param), arena);
                list_foreach(param_node, new_def.param_list)
                {
                    Param* param = (Param*)param_node->data;
                    list_foreach(struct_node, program->struct_list)
                    {
                        Struct* structure = (Struct*)struct_node->data;
                        if (structure->numeric)
                        {
                            Token new_name = token_copy(structure->name, arena);
                            Token numeric_struct_name = token_copy(structure->name, arena);
                            if (str_str(param->data_type.text, numeric_struct_name.text))
                            {
                                new_name = token_append(new_name, underscore, arena);
                                new_name = token_append(new_name, numeric_data_types[num_type], arena);
                                Token param_data_type = token_replace(param->data_type, numeric_struct_name.text, new_name.text, arena);
                                if (!token_equals(param->data_type, param_data_type.text))
                                    param->data_type = param_data_type;
                                else
                                    param->data_type = token_replace(param->data_type, numeric.text, numeric_data_types[num_type].text, arena);

                                param->name = token_replace(param->name, numeric_struct_name.text, new_name.text, arena);
                            }
                        }
                    }
                }

                // TODO(lucas): There are a lot of extra newlines inside function definitions.
                // TODO(lucas): A lot of the replacement code is very similar. Can this be pulled out?

                // NOTE(lucas): Check if any tokens inside the definition contain the numeric struct or keyword
                // and replace them appropriately
                new_def.token_list = list_copy(def->token_list, sizeof(Token), arena);
                list_foreach(token_node, new_def.token_list)
                {
                    Token* token = (Token*)token_node->data;
                    list_foreach(struct_node, program->struct_list)
                    {
                        Struct* structure = (Struct*)struct_node->data;
                        if (structure->numeric)
                        {
                            Token new_name = token_copy(structure->name, arena);
                            Token numeric_struct_name = token_copy(structure->name, arena);

                            if (str_str(token->text, numeric_struct_name.text))
                            {
                                new_name = token_append(new_name, underscore, arena);
                                new_name = token_append(new_name, numeric_data_types[num_type], arena);
                                Token new_token = token_replace(*token, numeric_struct_name.text, new_name.text, arena);
                                if (!token_equals(*token, new_token.text))
                                    *token = new_token;
                                else
                                    *token = token_replace(*token, numeric.text, numeric_data_types[num_type].text, arena);
                            }

                            // NOTE(lucas): Replace %num printf format specifier
                            if ((token->type == TOK_STRING) && (str_str(token->text, "%num") || (str_str(token->text, "%*num"))))
                            {
                                char* rep = "";
                                switch(num_type)
                                {
                                    // TODO(lucas): This is probably not very cross-platform
                                    // TODO(lucas): Floats: precision level, uppercase/lowercase,
                                    // scientific notation, shortest representation (Gg/Ee), hex float (Aa)
                                    case NUM_TYPE_U8:  rep = "hhu"; break;
                                    case NUM_TYPE_U16: rep = "hu"; break;
                                    case NUM_TYPE_U32: rep = "u"; break;
                                    case NUM_TYPE_U64: rep = "llu"; break;
                                    case NUM_TYPE_I8:  rep = "hhd"; break;
                                    case NUM_TYPE_I16: rep = "hd"; break;
                                    case NUM_TYPE_I32: rep = "d"; break;
                                    case NUM_TYPE_I64: rep = "lld"; break;
                                    case NUM_TYPE_F32: rep = "f"; break;
                                    case NUM_TYPE_F64: rep = "f"; break;
                                }
                                *token = token_replace(*token, "num", rep, arena);
                            }
                        }
                    }
                }
                function_def_write_to_file(dest, &new_def);
            }

            if (!numeric_found)
                function_def_write_to_file(dest, def);
        }

        fclose(out_h);
        fclose(out_c);
    }
}

void parse_file(Program* program, char* file_path)
{
    Parser parser = parser_init(file_path);

    char* out_file = str_until(file_path, ".", &parser.arena);

    b32 parsing = true;
    while (parsing)
    {
        Token token = token_get(&parser);
        switch (token.type)
        {
            case TOK_EOF: parsing = false; break;
            case TOK_UNKNOWN: break;

            case TOK_HASH:
            {
                Directive* directive = push_struct(&parser.arena, Directive); 
                *directive = parse_directive(&parser);
                list_push(&program->directive_list, directive, &parser.arena);
            } break;

            case TOK_IDENTIFIER:
            {
                if (token_equals(token, "struct"))
                {
                    Struct* structure = push_struct(&parser.arena, Struct);
                    *structure = parse_struct(&parser);
                    list_push(&program->struct_list, structure, &parser.arena);
                }
            } break;

            case TOK_TAG:
            {
                // TODO(lucas): I am not a fan of the begin/end tags.
                // This is not final by any means, just the easiest way I can think of
                // to do what I want for now.
                if (token_equals(token, "api_iface_begin"))
                    program->iface = parse_iface(&parser);
                    
                else if (token_equals(token, "api_impl_begin"))
                    program->impl = parse_impl(&parser);
            } break;

            default: break;
        }
    }

    write_output_file(out_file, program, &parser.arena);
}

int main(int argc, char* argv[])
{
    // TODO(lucas): Print program usage
    if (argc < 2)
    {
        log_fatal("Too few arguments.\n");
        return EXIT_FAILURE;
    }

    // TODO(lucas): Output directory option
    Program program = {0};
    for (usize i = 1; i < argc; ++i)
    {
        char* file_path = argv[i];
        if (str_len(file_path) > MAX_PATH_LEN)
            log_error("File path too long: %s.\n", file_path);

        parse_file(&program, file_path);

        list_clear(&program.directive_list);
    }
}
