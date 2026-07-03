package {
    import flash.display.Sprite;
    import flash.system.Worker;
    import flash.system.WorkerDomain;
    import flash.utils.ByteArray;

    public class Test extends Sprite {
        public function Test() {
            var primordial:Worker = Worker.current;

            // Unset key -> undefined.
            trace("unset = " + primordial.getSharedProperty("missing"));

            // Set/get scalar values.
            primordial.setSharedProperty("num", 42);
            trace("num = " + primordial.getSharedProperty("num"));

            primordial.setSharedProperty("str", "hello");
            trace("str = " + primordial.getSharedProperty("str"));

            // Overwrite.
            primordial.setSharedProperty("num", 99);
            trace("num overwritten = " + primordial.getSharedProperty("num"));

            // Objects are stored by reference (same instance comes back).
            var ba:ByteArray = new ByteArray();
            ba.writeUTFBytes("shared");
            primordial.setSharedProperty("ba", ba);
            trace("ba same object = " + (primordial.getSharedProperty("ba") === ba));

            // A separately created worker has its own independent store.
            var w:Worker = WorkerDomain.current.createWorker(new ByteArray());
            trace("w missing = " + w.getSharedProperty("num"));
            w.setSharedProperty("wkey", 7);
            trace("w wkey = " + w.getSharedProperty("wkey"));
            trace("primordial wkey = " + primordial.getSharedProperty("wkey"));
        }
    }
}
