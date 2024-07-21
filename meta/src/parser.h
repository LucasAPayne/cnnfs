#include "arena.h"

#include <string.h>

typedef struct Node Node;
struct Node
{
    void* data;
    Node* next;
};

typedef struct List
{
    Node* head;
} List;

#define list_foreach(it, list) \
    for (Node* it = list.head; it; it = it->next)

inline void list_push(List* list, void* new_data, Arena* arena)
{
    Node* new_node = push_struct(arena, Node);
    new_node->data = new_data;
    new_node->next = 0;

    if (!list->head)
        list->head = new_node;
    else
    {
        Node* it = list->head;
        while (it->next)
            it = it->next;

        it->next = new_node;
    }
}

inline List list_copy(List list, usize data_size, Arena* arena)
{
    ASSERT(list.head);
    List new_list = {0};

    list_foreach(it, list)
    {
        void* data = push_size(arena, data_size);
        memcpy(data, it->data, data_size);
        list_push(&new_list, data, arena);
    }

    return new_list;
}

inline void* list_at(List list, usize index)
{
    void* result = 0;
    Node* it = list.head;
    for (usize i = 0; it->next && i < index; ++i)
        it = it->next;

    if (it)
        result = it->data;
    
    return result;
}

// TODO(lucas): When I can delete from the middle of an arena,
// I need to actually clear the values of the list.
inline void list_clear(List* list)
{
    list->head = 0;
}

// TODO(lucas): Organize this enum and the token_get function a little better
typedef enum TokenType
{
    TOK_UNKNOWN = 0,
    TOK_DOT,
    TOK_ARROW,
    TOK_COMMA,
    TOK_COLON,
    TOK_SEMICOLON,
    TOK_QUESTION,
    TOK_PLUS,
    TOK_PLUS_EQ,
    TOK_MINUS,
    TOK_MINS_EQ,
    TOK_DIV,
    TOK_DIV_EQ,
    TOK_MOD,
    TOK_MOD_EQ,
    TOK_INCREMENT,
    TOK_DECREMENT,
    TOK_STAR,
    TOK_HASH,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_LT,
    TOK_LE,
    TOK_GT,
    TOK_GE,
    TOK_LSHIFT,
    TOK_LSHIFT_EQ,
    TOK_RSHIFT,
    TOK_RSHIFT_EQ,
    TOK_ASSIGN,
    TOK_EQ,
    TOK_NE,
    TOK_NOT,
    TOK_AND,
    TOK_OR,
    TOK_BIT_AND,
    TOK_BIT_AND_EQ,
    TOK_BIT_OR,
    TOK_BIT_OR_EQ,
    TOK_BIT_XOR,
    TOK_BIT_XOR_EQ,
    TOK_BIT_NOT,
    TOK_EOL,
    TOK_EOF,

    TOK_COMMENT,
    TOK_TAG,
    TOK_STRING,
    TOK_IDENTIFIER
} TokenType;

typedef struct Token
{
    TokenType type;
    usize length;
    char* text;
} Token;

typedef struct Tokenizer
{
    char* at;
} Tokenizer;

//
// NOTE(lucas): Language
//
typedef enum NumericType
{
    NUM_TYPE_NONE = 0,

    NUM_TYPE_U8,
    NUM_TYPE_U16,
    NUM_TYPE_U32,
    NUM_TYPE_U64,

    NUM_TYPE_I8,
    NUM_TYPE_I16,
    NUM_TYPE_I32,
    NUM_TYPE_I64,

    NUM_TYPE_F32,
    NUM_TYPE_F64,

    NUM_TYPE_COUNT
} NumericType;

typedef enum TagType
{
    TAG_UNKNOWN = 0,
    TAG_ARRAY
} TagType;

typedef struct Tag
{
    Token name;
    TagType type;
} Tag;

typedef struct Member
{
    Token name;
    Token data_type;
    Tag tag;
} Member;

typedef struct Struct
{
    Token name;
    b32 numeric;
    b32 written_to_file;
    usize numeric_index;
    List member_list;
} Struct;

// NOTE(lucas): Preprocessor directive
typedef struct Directive
{
    Token token;
} Directive;

/* NOTE(lucas): An API is made up of function declarations.
 * Functions declarations are made up of a name, return type, and parameter list.
 * A parameter is made up of a data type and a name
 * An implementation is made up function definitions
 * Function definitions are made up of the declaration info (parameters must be named)
 * and the function body.
 */
typedef struct Param
{
    Token name;
    Token data_type;
} Param;

typedef struct FunctionDecl
{
    Token name;
    Token return_type;
    Token keywords;
    List param_list;
} FunctionDecl;

typedef struct FunctionDef
{
    Token name;
    Token return_type;
    Token keywords;
    List token_list;
    List param_list;
} FunctionDef;

typedef struct Iface
{
    List decl_list;
} Iface;

typedef struct Impl
{
    List def_list;
} Impl;

typedef struct Program
{
    List directive_list;
    List struct_list;

    Iface iface;
    Impl impl;
} Program;

typedef struct Parser
{
    // TODO(lucas): Store current token so it doesn't
    // have to be passed to some funcitons?
    Tokenizer tokenizer;
    Arena arena;   
} Parser;
