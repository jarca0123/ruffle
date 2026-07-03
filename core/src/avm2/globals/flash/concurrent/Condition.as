package flash.concurrent {
    [API("684")]
    [Ruffle(InstanceAllocator)]
    public final class Condition {
        public function Condition(mutex:Mutex) {
            init(mutex);
        }

        public static native function get isSupported():Boolean;

        private native function init(mutex:Mutex):void;

        public native function notify():void;
        public native function notifyAll():void;
        public native function wait(timeout:Number = 4294967295):Boolean;
    }
}
