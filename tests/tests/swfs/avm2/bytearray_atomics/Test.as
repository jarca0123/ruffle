package {
    import flash.display.Sprite;
    import flash.utils.ByteArray;
    import flash.utils.Endian;

    public class Test extends Sprite {
        public function Test() {
            var ba:ByteArray = new ByteArray();
            ba.endian = Endian.LITTLE_ENDIAN;
            ba.length = 16;

            // Seed a known int at index 4.
            ba.position = 4;
            ba.writeInt(100);

            // CAS success: expected 100 matches -> swap to 200, returns old (100).
            trace("cas1 returned = " + ba.atomicCompareAndSwapIntAt(4, 100, 200));
            ba.position = 4;
            trace("value now = " + ba.readInt());

            // CAS failure: expected 999 does not match -> no swap, returns actual (200).
            trace("cas2 returned = " + ba.atomicCompareAndSwapIntAt(4, 999, 500));
            ba.position = 4;
            trace("value still = " + ba.readInt());

            // Out of range (14 + 4 > 16) -> RangeError.
            try {
                ba.atomicCompareAndSwapIntAt(14, 0, 1);
                trace("no error (unexpected)");
            } catch (e:*) {
                trace("range error caught");
            }

            // atomicCompareAndSwapLength success: expected 16 matches -> 32, returns 16.
            trace("caslen1 returned = " + ba.atomicCompareAndSwapLength(16, 32));
            trace("length now = " + ba.length);

            // atomicCompareAndSwapLength failure: length is 32, expected 16 -> no change, returns 32.
            trace("caslen2 returned = " + ba.atomicCompareAndSwapLength(16, 8));
            trace("length still = " + ba.length);
        }
    }
}
