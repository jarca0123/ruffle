// CrossBridge/Alchemy domain-memory intrinsics. The load/store intrinsics
// compile straight to AVM2 memory opcodes; `casi32` (atomic compare-and-swap)
// has no opcode, so it is a real native call.
package avm2.intrinsics.memory {
    public native function casi32(address:int, expectedValue:int, newValue:int):int;
    public native function mfence():void;
}
