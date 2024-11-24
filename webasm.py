# Copyright (c) 2024 Jake Ehrlich
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This file is very much under construction. The hope is that it acts
# as a future foundation for producing webasm binaries

from ctypes import Structure, POINTER, c_uint8, c_uint32, c_char_p, c_void_p
from enum import IntEnum

class ValType(IntEnum):
    i32 = 0x7F
    i64 = 0x7E
    f32 = 0x7D
    f64 = 0x7C
    v128 = 0x7B
    funcref = 0x70
    externref = 0x6F

class RefType(IntEnum):
    funcref = 0x70
    externref = 0x6F

class NumType(IntEnum):
    i32 = 0x7F
    i64 = 0x7E
    f32 = 0x7D
    f64 = 0x7C

class VecType(IntEnum):
    v128 = 0x7B

class Section(IntEnum):
    CUSTOM = 0
    TYPE = 1
    IMPORT = 2
    FUNCTION = 3
    TABLE = 4
    MEMORY = 5
    GLOBAL = 6
    EXPORT = 7
    START = 8
    ELEMENT = 9
    CODE = 10
    DATA = 11
    DATA_COUNT = 12

class FunctionType(Structure):
    _fields_ = [
        ("tag", c_uint8), # 0x60
        ("param_count", c_uint32),
        ("param_types", POINTER(c_uint8)),
        ("return_count", c_uint32),
        ("return_types", POINTER(c_uint8))
    ]

class TableType(Structure):
    _fields_ = [
        ("element_type", c_uint8),
        ("limits_flags", c_uint8),
        ("limits_min", c_uint32),
        ("limits_max", c_uint32)
    ]

class MemoryType(Structure):
    _fields_ = [
        ("limits_flags", c_uint8),
        ("limits_min", c_uint32),
        ("limits_max", c_uint32)
    ]

class GlobalType(Structure):
    _fields_ = [
        ("value_type", c_uint8),
        ("mutability", c_uint8)
    ]

class ImportEntry(Structure):
    _fields_ = [
        ("module_len", c_uint32),
        ("module_str", c_char_p),
        ("field_len", c_uint32),
        ("field_str", c_char_p),
        ("kind", c_uint8),
        ("type", c_void_p)
    ]

class ExportEntry(Structure):
    _fields_ = [
        ("field_len", c_uint32),
        ("field_str", c_char_p),
        ("kind", c_uint8),
        ("index", c_uint32)
    ]

class CodeEntry(Structure):
    _fields_ = [
        ("size", c_uint32),
        ("code", c_void_p)
    ]

class DataEntry(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("offset", c_void_p),
        ("size", c_uint32),
        ("data", c_char_p)
    ]

class ElementEntry(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("offset", c_void_p),
        ("func_count", c_uint32),
        ("func_indices", POINTER(c_uint32))
    ]

class WasmModule(Structure):
    _fields_ = [
        ("magic", c_uint32),
        ("version", c_uint32),
        ("type_count", c_uint32),
        ("types", POINTER(FunctionType)),
        ("import_count", c_uint32),
        ("imports", POINTER(ImportEntry)),
        ("function_count", c_uint32),
        ("functions", POINTER(c_uint32)),
        ("table_count", c_uint32),
        ("tables", POINTER(TableType)),
        ("memory_count", c_uint32),
        ("memories", POINTER(MemoryType)),
        ("global_count", c_uint32),
        ("globals", POINTER(GlobalType)),
        ("export_count", c_uint32),
        ("exports", POINTER(ExportEntry)),
        ("start_function", c_uint32),
        ("element_count", c_uint32),
        ("elements", POINTER(ElementEntry)),
        ("code_count", c_uint32),
        ("codes", POINTER(CodeEntry)),
        ("data_count", c_uint32),
        ("data", POINTER(DataEntry))
    ]

class WasmInstruction:
    # Control instructions
    UNREACHABLE = 0x00
    NOP = 0x01
    BLOCK = 0x02
    LOOP = 0x03
    IF = 0x04
    ELSE = 0x05
    END = 0x0B
    BR = 0x0C
    BR_IF = 0x0D
    BR_TABLE = 0x0E
    RETURN = 0x0F
    CALL = 0x10
    CALL_INDIRECT = 0x11

    # Reference instructions
    REF_NULL = 0xD0
    REF_IS_NULL = 0xD1
    REF_FUNC = 0xD2

    # Parametric instructions
    DROP = 0x1A
    SELECT = 0x1B
    SELECT_T = 0x1C

    # Variable instructions
    LOCAL_GET = 0x20
    LOCAL_SET = 0x21
    LOCAL_TEE = 0x22
    GLOBAL_GET = 0x23
    GLOBAL_SET = 0x24

    # Table instructions
    TABLE_GET = 0x25
    TABLE_SET = 0x26
    TABLE_INIT = 0xFC12
    ELEM_DROP = 0xFC13
    TABLE_COPY = 0xFC14
    TABLE_GROW = 0xFC15
    TABLE_SIZE = 0xFC16
    TABLE_FILL = 0xFC17

    # Memory instructions
    I32_LOAD = 0x28
    I64_LOAD = 0x29
    F32_LOAD = 0x2A
    F64_LOAD = 0x2B
    I32_LOAD8_S = 0x2C
    I32_LOAD8_U = 0x2D
    I32_LOAD16_S = 0x2E
    I32_LOAD16_U = 0x2F
    I64_LOAD8_S = 0x30
    I64_LOAD8_U = 0x31
    I64_LOAD16_S = 0x32
    I64_LOAD16_U = 0x33
    I64_LOAD32_S = 0x34
    I64_LOAD32_U = 0x35
    I32_STORE = 0x36
    I64_STORE = 0x37
    F32_STORE = 0x38
    F64_STORE = 0x39
    I32_STORE8 = 0x3A
    I32_STORE16 = 0x3B
    I64_STORE8 = 0x3C
    I64_STORE16 = 0x3D
    I64_STORE32 = 0x3E
    MEMORY_SIZE = 0x3F
    MEMORY_GROW = 0x40
    MEMORY_INIT = 0xFC08
    DATA_DROP = 0xFC09
    MEMORY_COPY = 0xFC0A
    MEMORY_FILL = 0xFC0B

    # Numeric instructions
    I32_CONST = 0x41
    I64_CONST = 0x42
    F32_CONST = 0x43
    F64_CONST = 0x44

    I32_EQZ = 0x45
    I32_EQ = 0x46
    I32_NE = 0x47
    I32_LT_S = 0x48
    I32_LT_U = 0x49
    I32_GT_S = 0x4A
    I32_GT_U = 0x4B
    I32_LE_S = 0x4C
    I32_LE_U = 0x4D
    I32_GE_S = 0x4E
    I32_GE_U = 0x4F

    I64_EQZ = 0x50
    I64_EQ = 0x51
    I64_NE = 0x52
    I64_LT_S = 0x53
    I64_LT_U = 0x54
    I64_GT_S = 0x55
    I64_GT_U = 0x56
    I64_LE_S = 0x57
    I64_LE_U = 0x58
    I64_GE_S = 0x59
    I64_GE_U = 0x5A

    F32_EQ = 0x5B
    F32_NE = 0x5C
    F32_LT = 0x5D
    F32_GT = 0x5E
    F32_LE = 0x5F
    F32_GE = 0x60

    F64_EQ = 0x61
    F64_NE = 0x62
    F64_LT = 0x63
    F64_GT = 0x64
    F64_LE = 0x65
    F64_GE = 0x66

    I32_CLZ = 0x67
    I32_CTZ = 0x68
    I32_POPCNT = 0x69
    I32_ADD = 0x6A
    I32_SUB = 0x6B
    I32_MUL = 0x6C
    I32_DIV_S = 0x6D
    I32_DIV_U = 0x6E
    I32_REM_S = 0x6F
    I32_REM_U = 0x70
    I32_AND = 0x71
    I32_OR = 0x72
    I32_XOR = 0x73
    I32_SHL = 0x74
    I32_SHR_S = 0x75
    I32_SHR_U = 0x76
    I32_ROTL = 0x77
    I32_ROTR = 0x78

    I64_CLZ = 0x79
    I64_CTZ = 0x7A
    I64_POPCNT = 0x7B
    I64_ADD = 0x7C
    I64_SUB = 0x7D
    I64_MUL = 0x7E
    I64_DIV_S = 0x7F
    I64_DIV_U = 0x80
    I64_REM_S = 0x81
    I64_REM_U = 0x82
    I64_AND = 0x83
    I64_OR = 0x84
    I64_XOR = 0x85
    I64_SHL = 0x86
    I64_SHR_S = 0x87
    I64_SHR_U = 0x88
    I64_ROTL = 0x89
    I64_ROTR = 0x8A

    F32_ABS = 0x8B
    F32_NEG = 0x8C
    F32_CEIL = 0x8D
    F32_FLOOR = 0x8E
    F32_TRUNC = 0x8F
    F32_NEAREST = 0x90
    F32_SQRT = 0x91
    F32_ADD = 0x92
    F32_SUB = 0x93
    F32_MUL = 0x94
    F32_DIV = 0x95
    F32_MIN = 0x96
    F32_MAX = 0x97
    F32_COPYSIGN = 0x98

    F64_ABS = 0x99
    F64_NEG = 0x9A
    F64_CEIL = 0x9B
    F64_FLOOR = 0x9C
    F64_TRUNC = 0x9D
    F64_NEAREST = 0x9E
    F64_SQRT = 0x9F
    F64_ADD = 0xA0
    F64_SUB = 0xA1
    F64_MUL = 0xA2
    F64_DIV = 0xA3
    F64_MIN = 0xA4
    F64_MAX = 0xA5
    F64_COPYSIGN = 0xA6

    I32_WRAP_I64 = 0xA7
    I32_TRUNC_F32_S = 0xA8
    I32_TRUNC_F32_U = 0xA9
    I32_TRUNC_F64_S = 0xAA
    I32_TRUNC_F64_U = 0xAB
    I64_EXTEND_I32_S = 0xAC
    I64_EXTEND_I32_U = 0xAD
    I64_TRUNC_F32_S = 0xAE
    I64_TRUNC_F32_U = 0xAF
    I64_TRUNC_F64_S = 0xB0
    I64_TRUNC_F64_U = 0xB1
    F32_CONVERT_I32_S = 0xB2
    F32_CONVERT_I32_U = 0xB3
    F32_CONVERT_I64_S = 0xB4
    F32_CONVERT_I64_U = 0xB5
    F32_DEMOTE_F64 = 0xB6
    F64_CONVERT_I32_S = 0xB7
    F64_CONVERT_I32_U = 0xB8
    F64_CONVERT_I64_S = 0xB9
    F64_CONVERT_I64_U = 0xBA
    F64_PROMOTE_F32 = 0xBB
    I32_REINTERPRET_F32 = 0xBC
    I64_REINTERPRET_F64 = 0xBD
    F32_REINTERPRET_I32 = 0xBE
    F64_REINTERPRET_I64 = 0xBF

    I32_EXTEND8_S = 0xC0
    I32_EXTEND16_S = 0xC1
    I64_EXTEND8_S = 0xC2
    I64_EXTEND16_S = 0xC3
    I64_EXTEND32_S = 0xC4

    def __init__(self, opcode, immediate=None):
        self.opcode = opcode
        self.immediate = immediate

class WasmFunction:
    def __init__(self, signature_type=None):
        self.signature = signature_type
        self.locals = []  # List of local variable types
        self.instructions = []  # List of WasmInstruction objects
        self.body_size = 0

    def add_local(self, val_type):
        self.locals.append(val_type)
        return len(self.locals) - 1

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def to_bytes(self):
        # Convert function to binary format
        local_bytes = []

        # Write local variable declarations
        if self.locals:
            local_bytes.extend(self._encode_unsigned_leb128(len(self.locals)))
            for local in self.locals:
                local_bytes.extend(self._encode_unsigned_leb128(1))  # Count
                local_bytes.append(local)  # Type

        # Write instructions
        body_bytes = []
        for inst in self.instructions:
            body_bytes.append(inst.opcode)
            if inst.immediate is not None:
                if isinstance(inst.immediate, int):
                    body_bytes.extend(self._encode_signed_leb128(inst.immediate))
                else:
                    body_bytes.extend(inst.immediate)

        # Add END opcode
        body_bytes.append(WasmInstruction.END)

        # Calculate body size
        total_size = len(local_bytes) + len(body_bytes)
        size_bytes = self._encode_unsigned_leb128(total_size)

        # Concatenate everything
        return bytes(size_bytes + local_bytes + body_bytes)

    def _encode_unsigned_leb128(self, value):
        # Encode unsigned LEB128
        result = []
        while True:
            byte = value & 0x7f
            value >>= 7
            if value:
                byte |= 0x80
            result.append(byte)
            if not value:
                break
        return result

    def _encode_signed_leb128(self, value):
        # Encode signed LEB128
        result = []
        while True:
            byte = value & 0x7f
            value >>= 7
            if (value == 0 and byte & 0x40 == 0) or (value == -1 and byte & 0x40 != 0):
                result.append(byte)
                break
            byte |= 0x80
            result.append(byte)
        return result
# Create a simple add function module
def create_add_module():
    # Create new module
    module = WasmModule()
    module.magic = 0x6D736100  # \0asm
    module.version = 0x1

    # Create function type (i32, i32) -> i32
    func_type = FunctionType()
    func_type.tag = 0x60
    func_type.param_count = 2
    func_type.param_types = (c_uint8 * 2)(ValType.i32, ValType.i32)
    func_type.return_count = 1
    func_type.return_types = (c_uint8 * 1)(ValType.i32)

    module.type_count = 1
    module.types = pointer(func_type)

    # Create function
    func = WasmFunction(0)  # Use type index 0

    # Add the function code
    func.add_instruction(WasmInstruction(WasmInstruction.LOCAL_GET, 0))
    func.add_instruction(WasmInstruction(WasmInstruction.LOCAL_GET, 1))
    func.add_instruction(WasmInstruction(WasmInstruction.I32_ADD))

    # Export the function
    export = ExportEntry()
    export.field_len = len("add")
    export.field_str = "add".encode('utf-8')
    export.kind = 0x00  # Function export
    export.index = 0

    module.export_count = 1
    module.exports = pointer(export)

    # Add code section
    code = CodeEntry()
    code.code = func.to_bytes()

    module.code_count = 1
    module.codes = pointer(code)

    # Write to file
    with open("add.wasm", "wb") as f:
        # Write magic and version
        f.write(module.magic.to_bytes(4, 'little'))
        f.write(module.version.to_bytes(4, 'little'))

        # Write type section
        f.write(bytes([Section.TYPE]))
        f.write(func_type.to_bytes())

        # Write function section
        f.write(bytes([Section.FUNCTION]))
        f.write(bytes([1, 0]))  # One function with type index 0

        # Write export section
        f.write(bytes([Section.EXPORT]))
        f.write(export.to_bytes())

        # Write code section
        f.write(bytes([Section.CODE]))
        f.write(code.to_bytes())
