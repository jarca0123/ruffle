package flash.concurrent {
    [API("684")]
    [Ruffle(InstanceAllocator)]
    public final class Mutex {
        public static native function get isSupported():Boolean;

        public function Mutex() {
        }

        public native function lock():void;
        public native function tryLock():Boolean;
        public native function unlock():void;
    }
}
